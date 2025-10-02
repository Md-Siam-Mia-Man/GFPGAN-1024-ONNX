import torch
from torch.nn import functional as F
from torch import nn
import math
from collections import OrderedDict
from noise import noise_dict


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode="down"):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == "down":
            self.scale_factor = 0.5
        elif mode == "up":
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        skip = self.skip(x)
        out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        return self.weight.repeat(batch, 1, 1, 1)


class GFPGAN_1024(nn.Module):
    def __init__(self, out_size=1024, num_style_feat=512, sft_half=True):
        super(GFPGAN_1024, self).__init__()

        self.log_size = int(math.log(out_size, 2))
        self.sft_half = sft_half
        self.num_style_feat = num_style_feat

        channels_unet = {
            "512": 32,
            "256": 64,
            "128": 128,
            "64": 256,
            "32": 256,
            "16": 256,
            "8": 256,
            "4": 256,
            "1024": 16,
        }

        stylegan_cm = 2
        stylegan_nf = 1.0
        channels_stylegan_base = {
            "4": int(512 * stylegan_nf),
            "8": int(512 * stylegan_nf),
            "16": int(512 * stylegan_nf),
            "32": int(512 * stylegan_nf),
            "64": int(256 * stylegan_cm * stylegan_nf),
            "128": int(128 * stylegan_cm * stylegan_nf),
            "256": int(64 * stylegan_cm * stylegan_nf),
            "512": int(32 * stylegan_cm * stylegan_nf),
        }

        self.conv_body_first = nn.Conv2d(3, channels_unet["512"], 1)

        self.conv_body_down = nn.ModuleList()
        unet_down_path_io_channels = [
            (channels_unet["512"], channels_unet["256"]),
            (channels_unet["256"], channels_unet["128"]),
            (channels_unet["128"], channels_unet["64"]),
            (channels_unet["64"], channels_unet["32"]),
            (channels_unet["32"], channels_unet["16"]),
            (channels_unet["16"], channels_unet["8"]),
            (channels_unet["8"], channels_unet["4"]),
        ]

        current_in_channels_for_resblock_def = channels_unet["512"]
        for i_rb_down, (in_ch_spec, out_ch_spec) in enumerate(
            unet_down_path_io_channels
        ):
            if current_in_channels_for_resblock_def != in_ch_spec:
                raise ValueError(
                    f"UNet down path definition error at ResBlock {i_rb_down}. Expected input {in_ch_spec}, got {current_in_channels_for_resblock_def} from previous block's output."
                )
            self.conv_body_down.append(ResBlock(in_ch_spec, out_ch_spec, mode="down"))
            current_in_channels_for_resblock_def = out_ch_spec

        self.final_conv = nn.Conv2d(
            current_in_channels_for_resblock_def, channels_unet["4"], 3, 1, 1
        )

        num_latents_stylegan = self.log_size * 2 - 2
        self.final_linear = nn.Linear(
            channels_unet["4"] * 4 * 4, (num_latents_stylegan - 2) * num_style_feat
        )
        self.final_extend_linear = nn.Linear(
            channels_unet["4"] * 4 * 4, 2 * num_style_feat
        )

        self.conv_body_up = nn.ModuleList()
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()

        unet_up_path_out_channels = [
            channels_unet["8"],
            channels_unet["16"],
            channels_unet["32"],
            channels_unet["64"],
            channels_unet["128"],
            channels_unet["256"],
            channels_unet["512"],
        ]

        current_in_channels_unet_up = channels_unet["4"]
        for i_up_main in range(len(unet_up_path_out_channels)):
            out_ch_unet_up_res = unet_up_path_out_channels[i_up_main]
            self.conv_body_up.append(
                ResBlock(current_in_channels_unet_up, out_ch_unet_up_res, mode="up")
            )
            current_in_channels_unet_up = out_ch_unet_up_res

            sft_cond_ch_source = out_ch_unet_up_res
            sft_cond_ch_target = sft_cond_ch_source
            if not self.sft_half:
                sft_cond_ch_target *= 2

            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(sft_cond_ch_source, sft_cond_ch_source, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(sft_cond_ch_source, sft_cond_ch_target, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(sft_cond_ch_source, sft_cond_ch_source, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(sft_cond_ch_source, sft_cond_ch_target, 3, 1, 1),
                )
            )

        self.final_body_up = ResBlock(
            channels_unet["512"], channels_unet["512"], mode="up"
        )
        _final_sft_cond_ch_hardcoded = 8
        self.final_scale = nn.Sequential(
            nn.Conv2d(channels_unet["512"], channels_unet["512"], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels_unet["512"], _final_sft_cond_ch_hardcoded, 3, 1, 1),
        )
        self.final_shift = nn.Sequential(
            nn.Conv2d(channels_unet["512"], channels_unet["512"], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels_unet["512"], _final_sft_cond_ch_hardcoded, 3, 1, 1),
        )

        self.stylegan_decoderdotconstant_input = ConstantInput(
            channels_stylegan_base["4"], size=4
        )
        self.stylegan_decoderdotstyle_conv1dotmodulated_convdotmodulation = nn.Linear(
            num_style_feat, channels_stylegan_base["4"], bias=True
        )
        self.stylegan_decoderdotstyle_conv1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(
                1, channels_stylegan_base["4"], channels_stylegan_base["4"], 3, 3
            )
            / math.sqrt(channels_stylegan_base["4"] * 3**2)
        )
        self.stylegan_decoderdotstyle_conv1dotweight = nn.Parameter(torch.zeros(1))
        self.stylegan_decoderdotstyle_conv1dotbias = nn.Parameter(
            torch.zeros(1, channels_stylegan_base["4"], 1, 1)
        )
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.stylegan_decoderdotto_rgb1dotmodulated_convdotmodulation = nn.Linear(
            num_style_feat, channels_stylegan_base["4"], bias=True
        )
        self.stylegan_decoderdotto_rgb1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, channels_stylegan_base["4"], 1, 1)
            / math.sqrt(channels_stylegan_base["4"] * 1**2)
        )
        self.stylegan_decoderdotto_rgb1dotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        _in_c_sg_base = channels_stylegan_base["4"]
        for i_conv_pair_sg_base in range(self.log_size - 3):
            i_res_log_sg_base = i_conv_pair_sg_base + 3
            _out_c_sg_base = channels_stylegan_base[f"{2**i_res_log_sg_base}"]
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2}dotmodulated_convdotmodulation",
                nn.Linear(num_style_feat, _in_c_sg_base, bias=True),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2}dotmodulated_convdotweight",
                nn.Parameter(
                    torch.randn(1, _out_c_sg_base, _in_c_sg_base, 3, 3)
                    / math.sqrt(_in_c_sg_base * 3**2)
                ),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2}dotweight",
                nn.Parameter(torch.zeros(1)),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2}dotbias",
                nn.Parameter(torch.zeros(1, _out_c_sg_base, 1, 1)),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2+1}dotmodulated_convdotmodulation",
                nn.Linear(num_style_feat, _out_c_sg_base, bias=True),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2+1}dotmodulated_convdotweight",
                nn.Parameter(
                    torch.randn(1, _out_c_sg_base, _out_c_sg_base, 3, 3)
                    / math.sqrt(_out_c_sg_base * 3**2)
                ),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2+1}dotweight",
                nn.Parameter(torch.zeros(1)),
            )
            setattr(
                self,
                f"stylegan_decoderdotstyle_convsdot{i_conv_pair_sg_base*2+1}dotbias",
                nn.Parameter(torch.zeros(1, _out_c_sg_base, 1, 1)),
            )
            setattr(
                self,
                f"stylegan_decoderdotto_rgbsdot{i_conv_pair_sg_base}dotmodulated_convdotmodulation",
                nn.Linear(num_style_feat, _out_c_sg_base, bias=True),
            )
            setattr(
                self,
                f"stylegan_decoderdotto_rgbsdot{i_conv_pair_sg_base}dotmodulated_convdotweight",
                nn.Parameter(
                    torch.randn(1, 3, _out_c_sg_base, 1, 1)
                    / math.sqrt(_out_c_sg_base * 1**2)
                ),
            )
            setattr(
                self,
                f"stylegan_decoderdotto_rgbsdot{i_conv_pair_sg_base}dotbias",
                nn.Parameter(torch.zeros(1, 3, 1, 1)),
            )
            _in_c_sg_base = _out_c_sg_base

        _final_csft_conv1_in_ch = 64
        _final_csft_conv1_out_ch = 16
        _final_csft_conv2_out_ch = 16
        self.stylegan_decoderdotfinal_conv1dotmodulated_convdotmodulation = nn.Linear(
            num_style_feat, _final_csft_conv1_in_ch, bias=True
        )
        self.stylegan_decoderdotfinal_conv1dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, _final_csft_conv1_out_ch, _final_csft_conv1_in_ch, 3, 3)
            / math.sqrt(_final_csft_conv1_in_ch * 3**2)
        )
        self.stylegan_decoderdotfinal_conv1dotweight = nn.Parameter(torch.zeros(1))
        self.stylegan_decoderdotfinal_conv1dotbias = nn.Parameter(
            torch.zeros(1, _final_csft_conv1_out_ch, 1, 1)
        )
        self.stylegan_decoderdotfinal_conv2dotmodulated_convdotmodulation = nn.Linear(
            num_style_feat, _final_csft_conv1_out_ch, bias=True
        )
        self.stylegan_decoderdotfinal_conv2dotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, _final_csft_conv2_out_ch, _final_csft_conv1_out_ch, 3, 3)
            / math.sqrt(_final_csft_conv1_out_ch * 3**2)
        )
        self.stylegan_decoderdotfinal_conv2dotweight = nn.Parameter(torch.zeros(1))
        self.stylegan_decoderdotfinal_conv2dotbias = nn.Parameter(
            torch.zeros(1, _final_csft_conv2_out_ch, 1, 1)
        )
        self.stylegan_decoderdotfinal_rgbdotmodulated_convdotmodulation = nn.Linear(
            num_style_feat, _final_csft_conv2_out_ch, bias=True
        )
        self.stylegan_decoderdotfinal_rgbdotmodulated_convdotweight = nn.Parameter(
            torch.randn(1, 3, _final_csft_conv2_out_ch, 1, 1)
            / math.sqrt(_final_csft_conv2_out_ch * 1**2)
        )
        self.stylegan_decoderdotfinal_rgbdotbias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def _stylegan_conv_block(
        self,
        x,
        latent_w,
        mod_lin,
        mod_w_param,
        noise_w_param,
        bias_p,
        upsample=False,
        noise_res=None,
        sft_scale=None,
        sft_shift=None,
    ):
        b = latent_w.size(0)

        _out_c_conv = int(mod_w_param.size(1))
        _in_c_conv_actual_from_weight = int(
            mod_w_param.size(2)
        )  # Channel size expected by conv weight's in_channels
        _in_c_conv_style_mod = int(
            mod_lin.out_features
        )  # Channel size output by modulation linear layer
        kernel_s = int(mod_w_param.size(3))

        # The style vector output from mod_lin must match the in_channels of the convolution weight.
        # This is usually `_in_c_conv_actual_from_weight`.
        style = mod_lin(latent_w).view(b, 1, _in_c_conv_style_mod, 1, 1)
        weight = mod_w_param * style

        is_torgb = False
        if _out_c_conv == 3 and kernel_s == 1:
            is_torgb = True

        if not is_torgb:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(b, _out_c_conv, 1, 1, 1)

        weight = weight.view(
            b * _out_c_conv, _in_c_conv_actual_from_weight, kernel_s, kernel_s
        )

        if upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        _b_x, _c_x, _h_x, _w_x = x.shape
        x_reshaped = x.view(1, _b_x * _c_x, _h_x, _w_x)

        calculated_padding = int(kernel_s // 2)

        out_conv = F.conv2d(x_reshaped, weight, padding=calculated_padding, groups=_b_x)

        out_conv = out_conv.view(_b_x, _out_c_conv, *out_conv.shape[2:4])

        if not is_torgb:
            out_conv = out_conv * (2**0.5)

        if noise_res is not None and noise_w_param is not None:
            current_noise_tensor = noise_dict[noise_res]
            if current_noise_tensor.device != out_conv.device:
                current_noise_tensor = current_noise_tensor.to(out_conv.device)
            if current_noise_tensor.size(0) != _b_x:
                current_noise_tensor = current_noise_tensor.repeat(_b_x, 1, 1, 1)
            out_conv = out_conv + noise_w_param * current_noise_tensor

        out_conv = out_conv + bias_p

        if not is_torgb:
            out_conv = self.activate(out_conv)
            if sft_scale is not None and sft_shift is not None:
                if self.sft_half:
                    split_size = int(out_conv.size(1) // 2)
                    out_same, out_sft_apply = torch.split(out_conv, split_size, dim=1)

                    if out_sft_apply.size(1) == sft_scale.size(1):
                        out_sft_apply = out_sft_apply * sft_scale + sft_shift
                    else:
                        out_sft_apply = out_sft_apply * sft_scale + sft_shift
                    out_conv = torch.cat([out_same, out_sft_apply], dim=1)
                else:
                    out_conv = out_conv * sft_scale + sft_shift
        return out_conv

    def _torgb_block(self, x, latent_w, mod_lin, mod_w_param, bias_p, skip_rgb=None):
        rgb = self._stylegan_conv_block(
            x, latent_w, mod_lin, mod_w_param, None, bias_p, upsample=False
        )
        if skip_rgb is not None:
            skip_rgb = F.interpolate(
                skip_rgb, scale_factor=2, mode="bilinear", align_corners=False
            )
            rgb = rgb + skip_rgb
        return rgb

    def forward(self, x):
        unet_skips = []
        feat_unet_current = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)

        for i_down in range(len(self.conv_body_down)):
            feat_unet_current = self.conv_body_down[i_down](feat_unet_current)
            unet_skips.insert(0, feat_unet_current)
        feat_unet_current = F.leaky_relu_(
            self.final_conv(feat_unet_current), negative_slope=0.2
        )

        style_code_main = self.final_linear(
            feat_unet_current.view(feat_unet_current.size(0), -1)
        )
        style_code_ext = self.final_extend_linear(
            feat_unet_current.view(feat_unet_current.size(0), -1)
        )
        style_code = torch.cat([style_code_main, style_code_ext], dim=1)
        num_latents_stylegan = self.log_size * 2 - 2  # 18 for 1024 output
        latent = style_code.view(
            style_code.size(0), num_latents_stylegan, self.num_style_feat
        )

        sft_conditions_from_unet = []
        for i_up_main in range(len(self.conv_body_up)):
            feat_unet_current = feat_unet_current + unet_skips[i_up_main]
            feat_unet_current = self.conv_body_up[i_up_main](feat_unet_current)
            scale = self.condition_scale[i_up_main](feat_unet_current)
            shift = self.condition_shift[i_up_main](feat_unet_current)
            sft_conditions_from_unet.extend([scale, shift])

        feat_for_final_sft = self.final_body_up(feat_unet_current)
        final_sft_scale_unet = self.final_scale(feat_for_final_sft)
        final_sft_shift_unet = self.final_shift(feat_for_final_sft)

        # StyleGAN Decoder
        sg_out = self.stylegan_decoderdotconstant_input(latent.shape[0])
        sg_out = self._stylegan_conv_block(
            sg_out,
            latent[:, 0],
            self.stylegan_decoderdotstyle_conv1dotmodulated_convdotmodulation,
            self.stylegan_decoderdotstyle_conv1dotmodulated_convdotweight,
            self.stylegan_decoderdotstyle_conv1dotweight,
            self.stylegan_decoderdotstyle_conv1dotbias,
            noise_res=4,
        )
        skip_rgb = self._torgb_block(
            sg_out,
            latent[:, 1],
            self.stylegan_decoderdotto_rgb1dotmodulated_convdotmodulation,
            self.stylegan_decoderdotto_rgb1dotmodulated_convdotweight,
            self.stylegan_decoderdotto_rgb1dotbias,
        )

        sg_latent_iter_idx = (
            1  # Start from 1 (index for latent vectors), as 0 and 1 are already used.
        )
        # This `sg_latent_iter_idx` refers to the latent index for the *first conv* of the current block.

        # Main StyleGAN loop: (log_size - 3) iterations. For 1024 output (log_size=10), this is 7 iterations.
        # These 7 iterations correspond to StyleGAN's 8x8 up to 512x512 feature maps.
        for i_sg_main_loop in range(self.log_size - 3):
            current_res_sg = 2 ** (i_sg_main_loop + 3)
            sft_scale = sft_conditions_from_unet[i_sg_main_loop * 2]
            sft_shift = sft_conditions_from_unet[i_sg_main_loop * 2 + 1]

            # First conv of the pair (upsamples)
            sg_out = self._stylegan_conv_block(
                sg_out,
                latent[:, sg_latent_iter_idx],
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2}dotmodulated_convdotmodulation",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2}dotmodulated_convdotweight",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2}dotweight",
                ),
                getattr(
                    self, f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2}dotbias"
                ),
                upsample=True,
                noise_res=current_res_sg,
                sft_scale=sft_scale,
                sft_shift=sft_shift,
            )

            # Second conv of the pair
            sg_out = self._stylegan_conv_block(
                sg_out,
                latent[:, sg_latent_iter_idx + 1],
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2+1}dotmodulated_convdotmodulation",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2+1}dotmodulated_convdotweight",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2+1}dotweight",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotstyle_convsdot{i_sg_main_loop*2+1}dotbias",
                ),
                noise_res=current_res_sg,
            )

            # ToRGB for this resolution
            skip_rgb = self._torgb_block(
                sg_out,
                latent[:, sg_latent_iter_idx + 2],
                getattr(
                    self,
                    f"stylegan_decoderdotto_rgbsdot{i_sg_main_loop}dotmodulated_convdotmodulation",
                ),
                getattr(
                    self,
                    f"stylegan_decoderdotto_rgbsdot{i_sg_main_loop}dotmodulated_convdotweight",
                ),
                getattr(self, f"stylegan_decoderdotto_rgbsdot{i_sg_main_loop}dotbias"),
                skip_rgb=skip_rgb,
            )

            sg_latent_iter_idx += 2  # Increment by 2 for the next pair of convs (as per original StyleGAN2GeneratorCSFT logic)

        # After loop, sg_latent_iter_idx = 1 (initial) + 7 * 2 = 15.
        # Latent indices used in loop: [1,2,3], [3,4,5], ..., [13,14,15]. Max index used is 15.

        # Final StyleGAN stage for 1024x1024 output
        # final_conv1 uses latent[:, sg_latent_iter_idx] which is latent[:, 15]
        sg_out = self._stylegan_conv_block(
            sg_out,
            latent[:, sg_latent_iter_idx],
            self.stylegan_decoderdotfinal_conv1dotmodulated_convdotmodulation,
            self.stylegan_decoderdotfinal_conv1dotmodulated_convdotweight,
            self.stylegan_decoderdotfinal_conv1dotweight,
            self.stylegan_decoderdotfinal_conv1dotbias,
            upsample=True,
            noise_res=1024,
            sft_scale=final_sft_scale_unet,
            sft_shift=final_sft_shift_unet,
        )

        # final_conv2 uses latent[:, sg_latent_iter_idx + 1] which is latent[:, 16]
        sg_out = self._stylegan_conv_block(
            sg_out,
            latent[:, sg_latent_iter_idx + 1],
            self.stylegan_decoderdotfinal_conv2dotmodulated_convdotmodulation,
            self.stylegan_decoderdotfinal_conv2dotmodulated_convdotweight,
            self.stylegan_decoderdotfinal_conv2dotweight,
            self.stylegan_decoderdotfinal_conv2dotbias,
            noise_res=1024,
        )

        # final_rgb uses latent[:, sg_latent_iter_idx + 2] which is latent[:, 17]
        image = self._torgb_block(
            sg_out,
            latent[:, sg_latent_iter_idx + 2],
            self.stylegan_decoderdotfinal_rgbdotmodulated_convdotmodulation,
            self.stylegan_decoderdotfinal_rgbdotmodulated_convdotweight,
            self.stylegan_decoderdotfinal_rgbdotbias,
            skip_rgb=skip_rgb,
        )
        # Max latent index used: 17. Total latents needed: 0 to 17 (18 latents).
        # `num_latents_stylegan` is 18. This matches.
        return image
