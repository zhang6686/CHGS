import argparse
import math
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch

from dataloaders.data_tools import joints_list
from models import DiffGestureGenerator
from models.vq.model import RVQVAE
from utils import other_tools_hf
from utils import rotation_conversions as rc
from utils_1 import NullableArgs

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class Demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        model_path, exp_name = self._get_model_path(args.exp_name, args.iter)
        self.exp_name = exp_name
        self.iter = args.iter
        model_data = torch.load(model_path, map_location=self.device)
        self.model_args = NullableArgs(model_data['args'])
        self.model = DiffGestureGenerator(self.model_args, self.device)
        model_data['model'].pop('denoising_net.TE.pe')
        self.model.load_state_dict(model_data['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.gt_npz_dir = Path(args.gt_npz_dir) if args.gt_npz_dir else None
        self.smplx_model_dir = self._resolve_smplx_model_dir(args.smplx_model_dir)
        self.result_dir = Path(args.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.ori_joint_list = joints_list["beat_smplx_joints"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_upper[end - start:end] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_hands[end - start:end] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_lower[end - start:end] = 1

        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        self.vq_type = "rvqvae"

        if self.vq_type == "rvqvae":
            self.args.num_quantizers = 6
            self.args.shared_codebook = False
            self.args.quantize_dropout_prob = 0.2
            self.args.mu = 0.99

            self.args.nb_code = 512
            self.args.code_dim = 512
            self.args.down_t = 2
            self.args.stride_t = 2
            self.args.width = 512
            self.args.depth = 3
            self.args.dilation_growth_rate = 3
            self.args.vq_act = "relu"
            self.args.vq_norm = None

            self.dim_pose = 78
            self.args.body_part = "upper"
            self.vq_model_upper = RVQVAE(
                self.args,
                self.dim_pose,
                self.args.nb_code,
                self.args.code_dim,
                self.args.code_dim,
                self.args.down_t,
                self.args.stride_t,
                self.args.width,
                self.args.depth,
                self.args.dilation_growth_rate,
                self.args.vq_act,
                self.args.vq_norm,
            )

            self.dim_pose = 180
            self.args.body_part = "hands"
            self.vq_model_hands = RVQVAE(
                self.args,
                self.dim_pose,
                self.args.nb_code,
                self.args.code_dim,
                self.args.code_dim,
                self.args.down_t,
                self.args.stride_t,
                self.args.width,
                self.args.depth,
                self.args.dilation_growth_rate,
                self.args.vq_act,
                self.args.vq_norm,
            )

            self.dim_pose = 54
            self.args.use_trans = True
            if self.args.use_trans:
                self.dim_pose = 57
                self.args.vqvae_lower_path = './ckpt/beatx2_rvqvae/RVQVAE_lower_trans/net_300000_1.pth'
            self.args.body_part = "lower"
            self.vq_model_lower = RVQVAE(
                self.args,
                self.dim_pose,
                self.args.nb_code,
                self.args.code_dim,
                self.args.code_dim,
                self.args.down_t,
                self.args.stride_t,
                self.args.width,
                self.args.depth,
                self.args.dilation_growth_rate,
                self.args.vq_act,
                self.args.vq_norm,
            )
            self.args.vqvae_upper_path = './ckpt/beatx2_rvqvae/RVQVAE_upper/net_300000_1.pth'
            self.args.vqvae_hands_path = './ckpt/beatx2_rvqvae/RVQVAE_hands/net_300000_1.pth'
            self.args.vqvae_lower_path = './ckpt/beatx2_rvqvae/RVQVAE_lower_trans/net_300000_1.pth'
            self.vq_model_upper.load_state_dict(torch.load(self.args.vqvae_upper_path, map_location=self.device)['net'])
            self.vq_model_hands.load_state_dict(torch.load(self.args.vqvae_hands_path, map_location=self.device)['net'])
            self.vq_model_lower.load_state_dict(torch.load(self.args.vqvae_lower_path, map_location=self.device)['net'])

            self.vqvae_latent_scale = 5

            self.vq_model_upper.eval().to(self.device)
            self.vq_model_hands.eval().to(self.device)
            self.vq_model_lower.eval().to(self.device)

        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.args.mean_pose_path = './mean_std/beatx_2_330_mean.npy'
        self.args.std_pose_path = './mean_std/beatx_2_330_std.npy'
        self.mean = np.load(self.args.mean_pose_path)
        self.std = np.load(self.args.std_pose_path)

        if self.args.use_trans:
            self.args.mean_trans_path = './mean_std/beatx_2_trans_mean.npy'
            self.args.std_trans_path = './mean_std/beatx_2_trans_std.npy'
            self.trans_mean = np.load(self.args.mean_trans_path)
            self.trans_std = np.load(self.args.std_trans_path)
            self.trans_mean = torch.from_numpy(self.trans_mean).to(self.device)
            self.trans_std = torch.from_numpy(self.trans_std).to(self.device)

        self.render_args = argparse.Namespace(
            debug=False,
            render_video_fps=30,
            render_video_width=1920,
            render_video_height=720,
            render_concurrent_num=max(1, (os.cpu_count() or 1) // 2),
            render_tmp_img_filetype="bmp",
        )

        upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.upper_body_mask = []
        for joint_idx in upper_joints:
            self.upper_body_mask.extend([joint_idx * 6 + offset for offset in range(6)])

        hand_joints = list(range(25, 55))
        self.hands_body_mask = []
        for joint_idx in hand_joints:
            self.hands_body_mask.extend([joint_idx * 6 + offset for offset in range(6)])

        lower_joints = [0, 1, 2, 4, 5, 7, 8, 10, 11]
        self.lower_body_mask = []
        for joint_idx in lower_joints:
            self.lower_body_mask.extend([joint_idx * 6 + offset for offset in range(6)])

        self.mean_upper = self.mean[self.upper_body_mask]
        self.mean_hands = self.mean[self.hands_body_mask]
        self.mean_lower = self.mean[self.lower_body_mask]
        self.std_upper = self.std[self.upper_body_mask]
        self.std_hands = self.std[self.hands_body_mask]
        self.std_lower = self.std[self.lower_body_mask]

        self.mean_upper = torch.from_numpy(self.mean_upper).to(self.device)
        self.mean_hands = torch.from_numpy(self.mean_hands).to(self.device)
        self.mean_lower = torch.from_numpy(self.mean_lower).to(self.device)
        self.std_upper = torch.from_numpy(self.std_upper).to(self.device)
        self.std_hands = torch.from_numpy(self.std_hands).to(self.device)
        self.std_lower = torch.from_numpy(self.std_lower).to(self.device)

    def infer_from_file(self, audio_path, out_path, cfg_scale=1.15):
        rec_latent_upper, rec_latent_hands, rec_latent_lower = self.infer_coeffs(audio_path, cfg_scale=cfg_scale)

        rec_upper = self.vq_model_upper.latent2origin(rec_latent_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_latent_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_latent_lower)[0]

        rec_trans_v = rec_lower[..., -3:]
        rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
        rec_trans = torch.zeros_like(rec_trans_v)
        rec_trans = torch.cumsum(rec_trans_v, dim=-2)
        rec_trans[..., 1] = rec_trans_v[..., 1]
        rec_lower = rec_lower[..., :-3]

        rec_upper = rec_upper * self.std_upper + self.mean_upper
        rec_hands = rec_hands * self.std_hands + self.mean_hands
        rec_lower = rec_lower * self.std_lower + self.mean_lower

        def inverse_selection_tensor(filtered_t, selection_array, total):
            selection_array = torch.from_numpy(selection_array).to(self.device)
            original_shape_t = torch.zeros((total, 165), device=self.device)
            selected_indices = torch.where(selection_array == 1)[0]
            for idx in range(total):
                original_shape_t[idx, selected_indices] = filtered_t[idx]
            return original_shape_t

        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]

        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs * n)

        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs * n)

        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs * n)

        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover

        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_trans_np = rec_trans.detach().cpu().numpy()
        rec_trans_np = np.squeeze(rec_trans_np, axis=0)

        audio_stem = Path(audio_path).stem
        beta_source_npz_path = Path("demo/result_vq/2_scott_0_3_3.npz")
        beta_source_npz = np.load(beta_source_npz_path, allow_pickle=True)
        n_frames = rec_pose_np.shape[0]
        rec_pose_np = rec_pose_np[:n_frames]
        rec_trans_np = rec_trans_np[:n_frames]
        expression_dim = 100
        if "expressions" in beta_source_npz.files and beta_source_npz["expressions"].ndim == 2:
            expression_dim = beta_source_npz["expressions"].shape[1]
        zero_expressions = np.zeros((n_frames, expression_dim), dtype=np.float32)
        save_path = self.result_dir / f'{audio_stem}_vq.npz'
        np.savez(
            str(save_path),
            betas=beta_source_npz["betas"],
            poses=rec_pose_np,
            expressions=zero_expressions,
            trans=np.zeros_like(rec_trans_np),
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate=30,
        )
        results_save_path = str(self.result_dir) + '/'

        render_vid_path = other_tools_hf.render_one_sequence_no_gt(
            str(save_path),
            results_save_path,
            str(audio_path),
            str(self.smplx_model_dir),
            use_matplotlib=False,
            args=self.render_args,
        )
        if out_path is not None:
            import shutil
            shutil.move(render_vid_path, str(out_path))
            print(f"Video saved to {out_path}")

    @torch.no_grad()
    def infer_coeffs(self, audio, cfg_scale=1.15):
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=16000, mono=True)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        assert audio.ndim == 1, 'Audio must be 1D tensor.'
        audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        audio_unit = 16000. / 30
        round_l = 128 - 16
        clip_len = int(len(audio) / 16000 * 30)
        roundt = math.floor(clip_len / round_l)
        if roundt < 1:
            raise ValueError('Audio is too short for 112-frame sampling.')
        length_n = roundt * round_l

        audio = audio[:round(audio_unit * length_n)]

        rec_all_upper = []
        rec_all_hands = []
        rec_all_lower = []
        prev_motion_feat = None
        noise = None
        prev_audio_feat = None

        for i in range(roundt):
            indicator = torch.ones((1, 28), device=self.device)
            audio_in = audio[round(i * audio_unit * round_l):round((i + 1) * audio_unit * round_l)].unsqueeze(0)

            if i == 0:
                motion_feat, noise, prev_audio_feat = self.model.sample(
                    audio_in,
                    indicator=indicator, cfg_mode=None,
                    cfg_cond=None, cfg_scale=cfg_scale,
                    dynamic_threshold=None)
            else:
                motion_feat, noise, prev_audio_feat = self.model.sample(
                    audio_in,
                    prev_motion_feat, prev_audio_feat, noise,
                    indicator=indicator, cfg_mode=None,
                    cfg_cond=None, cfg_scale=cfg_scale,
                    dynamic_threshold=None)

            prev_motion_feat = motion_feat[:, -4:].clone()
            prev_audio_feat = prev_audio_feat[:, -4:]
            motion_feat = motion_feat[:, 4:]

            rec_latent_upper = motion_feat[..., :512]
            rec_latent_hands = motion_feat[..., 512:1024]
            rec_latent_lower = motion_feat[..., 1024:1536]
            rec_all_upper.append(rec_latent_upper)
            rec_all_hands.append(rec_latent_hands)
            rec_all_lower.append(rec_latent_lower)

        motion_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        motion_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        motion_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale
        return motion_upper, motion_hands, motion_lower

    @staticmethod
    def _get_model_path(exp_name, iteration):
        exp_root_dir = Path(__file__).parent / 'experiments/DPT'
        exp_dir = exp_root_dir / exp_name
        if not exp_dir.exists():
            exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
        model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
        return model_path, exp_dir.relative_to(exp_root_dir)

    def _resolve_gt_npz_path(self, audio_path):
        audio_path = Path(audio_path)
        audio_stem = audio_path.stem
        candidates = []

        if self.gt_npz_dir is not None:
            candidates.append(self.gt_npz_dir / f"{audio_stem}.npz")
        if audio_path.parent.name == "wave16k":
            candidates.append(audio_path.parent.parent / "smplxflame_30" / f"{audio_stem}.npz")
        candidates.append(audio_path.with_suffix(".npz"))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Missing reference npz file for {audio_path}. "
            f"Tried: {searched}. "
            "Pass --gt_npz_dir explicitly if your npz files live elsewhere."
        )

    @staticmethod
    def _resolve_smplx_model_dir(model_dir):
        candidates = []
        if model_dir:
            candidates.append(Path(model_dir))
        candidates.extend([
            Path("smplx_models"),
            Path(__file__).parent / "smplx_models",
        ])

        for candidate in candidates:
            if (candidate / "smplx" / "SMPLX_NEUTRAL_2020.npz").exists():
                return candidate

        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Cannot find SMPL-X model files. "
            f"Tried: {searched}. "
            "Pass --smplx_model_dir explicitly."
        )


def main(args):
    demo_app = Demo(args)
    if args.mode == 'interactive':
        try:
            while True:
                audio = input('Enter audio file path: ')
                scale = float(input('Enter guiding scale (default: 1.15): ') or 1.15)
                output = input('Enter output file path: ')
                print('Generating...')
                demo_app.infer_from_file(audio, output, cfg_scale=scale)
                print('Done.\n')
        except KeyboardInterrupt:
            print()
            exit(0)
    else:
        demo_app.infer_from_file(args.audio, args.output, cfg_scale=args.cfg_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Speech-driven gesture demo with VQ motion decoding'
    )

    # Model
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--iter', type=int, default=1000000, help='number of iterations')

    # Runtime
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'interactive'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cfg_scale', type=float, default=1.15, help='classifier-free guidance scale')
    parser.add_argument('--gt_npz_dir', type=Path, default=None,
                        help='directory containing reference npz files with expressions/betas')
    parser.add_argument('--smplx_model_dir', type=Path, default=None,
                        help='directory containing SMPL-X model files for rendering')
    parser.add_argument('--result_dir', type=Path, default=Path('demo/result_vq'),
                        help='directory for intermediate render files')
    parser.add_argument('--n_repetitions', '-n', type=int, default=1, help=argparse.SUPPRESS)

    args = parser.parse_known_args()[0]
    if args.mode != 'interactive':
        parser.add_argument('--audio', '-a', type=Path, required=True, help='path of the input audio signal')
        parser.add_argument('--output', '-o', type=Path, required=True, help='path of the rendered video')

    args = parser.parse_args()
    main(args)
