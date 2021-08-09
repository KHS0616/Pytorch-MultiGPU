"""
Pytorch-MultiGPU 학습 예제 코드

Writer : KHS0616
Last Update : 2021-07-21
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse

from random import Random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.loss import VGGLoss, GANLoss
from models.models import Generator, Discriminator
from utils import AverageMeter, ProgressMeter, calc_psnr, calc_ssim, calc_lpips, preprocess
from utils import check_image_file
from dataset import Dataset

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
import torchvision.utils as vutils
# from lpips import LPIPS
from PIL import Image
from tensorboardX import SummaryWriter

import math

"""
python3 train_dist.py --train-file ../Image/DIV2K_train_HR/ --eval-file ../Image/DIV2K_vaild_HR/ --outputs-dir weights --scale 4 --pretrained-net weights/BSRNet.pth
nohup python3 train_dist.py --train-file ../Image/DIV2K_train_HR/ --eval-file ../Image/DIV2K_vaild_HR/ --outputs-dir weights --scale 4 --pretrained-net weights/BSRNet.pth &
"""

# 테스트 이미지 경로 설정
test_image_path = 'examples/0001.png'
# 테스트 이미지 불러오기
test_image = Image.open(test_image_path).convert('RGB')
# 테스트 이미지 전처리
test_image = preprocess(test_image)

def setArgparse():
    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--pretrained-net', type=str, default='BSRNet.pth')
    parser.add_argument('--gan-lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=288)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--resume-g', type=str, default='generator.pth')
    parser.add_argument('--resume-d', type=str, default='discriminator.pth')
    
    parser.add_argument('--dist', action="store_true")
    args = parser.parse_args()
    return args

def average_gradients(model):
    """ 변화도 평균 계산하기 """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

class DataPartitioner(object):
    """ 데이터 셋 분할 클래스 """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return self.partitions[partition]

class Train():
    def __init__(self, args):
        """ BSRGAN 학습 클래스 초기화 메소드 """
        # 사용할 GPU PCI 번호 리스트로 작성
        self.gpu_ids = [2, 3]

        # Argparse 값을 변수에 할당
        self.opt = args

        # 학습 결과를 저장할 경로 확인 및 생성
        self.opt.outputs_dir = os.path.join(args.outputs_dir,  f"BSRGAN_x{args.scale}")
        if not os.path.exists(self.opt.outputs_dir):
            os.makedirs(self.opt.outputs_dir)

        # Train 데이터 리스트 및 Eval 데이터 로더 생성
        self.setDataset()

    def setDataset(self):
        """ Train 데이터 리스트, Eval 데이터 로더 생성 메소드"""
        # 전체 Train 데이터 리스트를 변수에 저장
        self.train_dataset = [os.path.join(self.opt.train_file, x) for x in os.listdir(self.opt.train_file) if check_image_file(x)]

        # Eval 데이터 셋 및 데이터 로더 생성
        self.eval_dataset = Dataset(self.opt.eval_file, self.opt.patch_size, self.opt.scale)
        self.eval_dataloader = DataLoader(
                                    dataset=self.eval_dataset, 
                                    batch_size=self.opt.batch_size,
                                    shuffle=False,
                                    num_workers=self.opt.num_workers,
                                    drop_last=True,
                                    pin_memory=True
                                    )
    
    def init_process(self, rank, size, backend="gloo"):
        """ 분산 학습을 위한 프로세스 생성 함수 """
        # 분산 학습을 진행하는 IP 주소, 포트번호 설정
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        # Model Loss 공유를 위하여 그룹 생성
        dist.init_process_group(backend, rank=rank, world_size=size)

        # 학습 프로세스 실행
        self.run(rank, size)

    def start(self):
        """ 분산 학습을 시작하는 메소드 """
        # spawn 방식으로 프로세스 설정 - 현재 미 사용
        # mp.set_start_method("spawn")

        # GPU 개수 등의 정보 저장
        size = len(self.gpu_ids)
        processes = []

        # 데이터 셋 설정
        partition_sizes = [1.0 / size for _ in range(size)]
        self.partition = DataPartitioner(self.train_dataset, partition_sizes)

        # 프로세스 시작
        for rank in range(size):
            p = mp.Process(target=self.init_process, args=(rank, size))
            p.start()
            processes.append(p)            

        # 프로세스가 절차를 완료하거나 오류가 발생하면 안전하게 종료
        for p in processes:
            p.join()

    def run(self, rank, size):
        """ 멀티 프로세스를 이용한 스레드 학습 메소드 """
        # 텐서보드 설정
        if rank == 0:
            writer = SummaryWriter()

        # CUDA GPU 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids[rank])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 학습 데이터 리스트 가져오기 및 데이터 셋 설정
        dataindexes = self.partition.use(rank)
        datalist = [x for i, x in enumerate(self.train_dataset) if i in dataindexes]
        train_dataset = Dataset(images_dir="", image_size=self.opt.patch_size, upscale_factor=self.opt.scale, dist_status=True, dist_data_list=datalist)
        train_dataloader = DataLoader(
                                dataset=train_dataset,
                                batch_size=self.opt.batch_size,
                                shuffle=True,
                                num_workers=self.opt.num_workers,
                                pin_memory=True
                            )

        # Generator, Discriminator 모델 생성
        generator = Generator(scale_factor=self.opt.scale).to(device)
        discriminator = Discriminator().to(device)

        # L1 Loss, VGG19 Loss, Patch GAN Loss 생성
        pixel_criterion = nn.L1Loss().to(device)
        content_criterion = VGGLoss().to(device)
        adversarial_criterion = GANLoss().to(device)

        # Generator, Discriminator Optimizer 생성
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.opt.gan_lr, betas=(0.9, 0.999))
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.opt.gan_lr, betas=(0.9, 0.999))

        # Generator, Discriminator Scheduler 생성
        interval_epoch = math.ceil(self.opt.num_epochs // 8)
        epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
        discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=epoch_indices, gamma=0.5)
        generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=epoch_indices, gamma=0.5)
        scaler = amp.GradScaler()

        # 학습 관련 Epoch 설정
        total_epoch = self.opt.num_epochs
        g_epoch = 0
        d_epoch = 0
        best_lpips = 0

        # 기존에 학습된 Weight 파일이 있으면 불러오기
        if os.path.exists(self.opt.resume_g) and os.path.exists(self.opt.resume_d):
            # Generator Weight 불러오기
            checkpoint_g = torch.load(self.opt.resume_g)
            generator.load_state_dict(checkpoint_g['model_state_dict'])
            g_epoch = checkpoint_g['epoch'] + 1
            generator_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])

            # Discriminator Weight 불러오기
            checkpoint_d = torch.load(self.opt.resume_d)
            discriminator.load_state_dict(checkpoint_d['model_state_dict'])
            d_epoch = checkpoint_g['epoch'] + 1
            discriminator_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])
        elif os.path.exists(self.opt.pretrained_net):
            # BSRNet Weight 불러오기
            state_dict = generator.state_dict()
            for n, p in torch.load(self.opt.pretrained_net,map_location=device).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
        else:
            raise RuntimeError("You need pre-trained BSRGAN.pth or generator & discriminator")

        
        # 성능 측정을 위한 LPIPS 메트릭스 설정
        # lpips_metrics = LPIPS(net='vgg').to(device)

        # 학습 및 테스트 시작
        for epoch in range(g_epoch, total_epoch):
            # Generator, Discriminator 학습 모드로 변경
            generator.train()
            discriminator.train()

            """ Losses average meter 설정 """
            d_losses = AverageMeter(name="D Loss", fmt=":.6f")
            g_losses = AverageMeter(name="G Loss", fmt=":.6f")
            pixel_losses = AverageMeter(name="Pixel Loss", fmt=":6.4f")
            content_losses = AverageMeter(name="Content Loss", fmt=":6.4f")
            adversarial_losses = AverageMeter(name="adversarial losses", fmt=":6.4f")
            
            """ 모델 평가 measurements 설정 """
            psnr = AverageMeter(name="PSNR", fmt=":.6f")
            lpips = AverageMeter(name="LPIPS", fmt=":.6f")
            ssim = AverageMeter(name="SSIM", fmt=":.6f")

            """ progress meter 설정 """
            progress = ProgressMeter(
                num_batches=len(self.eval_dataloader)-1,
                meters=[psnr, lpips, ssim, d_losses, g_losses, pixel_losses, content_losses, adversarial_losses],
                prefix=f"Epoch: [{epoch}]"
            )
            
            """  트레이닝 Epoch 시작 """
            for i, (lr, hr) in enumerate(train_dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                batch_size = lr.size(0)

                # 리얼 라벨 1, 생선된 가짜 라벨 0
                real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).to(device)
                fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).to(device)

                # Discriminator 변화량 초기화
                discriminator_optimizer.zero_grad()

                # Discriminator Loss 생성
                with amp.autocast():
                    preds = generator(lr)

                    real_output = discriminator(hr)
                    d_loss_real = adversarial_criterion(real_output, True)

                    fake_output = discriminator(preds.detach())
                    d_loss_fake = adversarial_criterion(fake_output, False)

                    d_loss = (d_loss_real + d_loss_fake) / 2

                # Discriminator Loss 업데이트
                scaler.scale(d_loss).backward()
                average_gradients(discriminator)
                scaler.step(discriminator_optimizer)
                scaler.update()

                # Generator 변화량 초기화
                generator_optimizer.zero_grad()

                # Generator Loss 생성
                with amp.autocast():
                    preds = generator(lr)
                    real_output = discriminator(hr.detach())
                    fake_output = discriminator(preds)
                    pixel_loss = pixel_criterion(preds, hr.detach())
                    content_loss = content_criterion(preds, hr.detach())
                    adversarial_loss = adversarial_criterion(fake_output, True)

                    g_loss = (1 * pixel_loss) + (1 * content_loss) + (0.1 * adversarial_loss)

                # Generator Loss 업데이트
                scaler.scale(g_loss).backward()
                average_gradients(generator)
                scaler.step(generator_optimizer)
                scaler.update()

                generator.zero_grad()

                d_losses.update(d_loss.item(), lr.size(0))
                g_losses.update(g_loss.item(), lr.size(0))
                pixel_losses.update(pixel_loss.item(), lr.size(0))
                content_losses.update(content_loss.item(), lr.size(0))
                adversarial_losses.update(adversarial_loss.item(), lr.size(0))

            # Scheduler 업데이트
            discriminator_scheduler.step()
            generator_scheduler.step()
            print("Process ID : ", rank, " Loss : ", g_loss, " ", d_loss)

            # 1 Epoch 마다 텐서보드 업데이트
            if rank == 0:
                writer.add_scalar("d_Loss/train", d_losses.avg, epoch)
                writer.add_scalar("g_Loss/train", g_losses.avg, epoch)
                writer.add_scalar("pixel_losses/train", pixel_losses.avg, epoch)
                writer.add_scalar("adversarial_losses/train", content_losses.avg, epoch)
                writer.add_scalar("adversarial_losses/train", adversarial_losses.avg, epoch)

            """  테스트 Epoch 시작 """
            generator.eval()
            with torch.no_grad():
                for i, (lr, hr) in enumerate(self.eval_dataloader):
                    lr = lr.to(device)
                    hr = hr.to(device)
                    preds = generator(lr)
                    psnr.update(calc_psnr(preds, hr), len(lr))
                    ssim.update(calc_ssim(preds, hr), len(lr))

                    if i == len(self.eval_dataset)//self.opt.batch_size:
                        progress.display(i)
            if rank == 0:
                writer.add_scalar("psnr/test", psnr.avg, epoch)
                writer.add_scalar("ssim/test", ssim.avg, epoch)
                writer.add_scalar("lpips/test", lpips.avg, epoch)
            
            """  Best 모델 저장 """
            if lpips.avg < best_lpips:
                best_lpips = lpips.avg
                torch.save(
                    generator.state_dict(), os.path.join(self.opt.outputs_dir, 'best_g.pth')
                )

            """ Discriminator 모델 저장 """
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': discriminator_optimizer.state_dict(),
                }, os.path.join(self.opt.outputs_dir, 'd_epoch_{}.pth'.format(epoch))
            )
            """ Generator 모델 저장 """
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': generator_optimizer.state_dict(),
                    'best_lpips': best_lpips,
                }, os.path.join(self.opt.outputs_dir, 'g_epoch_{}.pth'.format(epoch))
            )
            """ 나비 이미지 테스트 """
            with torch.no_grad():
                lr = test_image.to(device)
                preds = generator(lr)
                vutils.save_image(preds.detach(), os.path.join(self.opt.outputs_dir, f"BSRGAN_{epoch}.jpg"))
        # 텐서보드 종료
        if rank == 0:
            writer.close()

if __name__ == '__main__':
    args = setArgparse()
    t = Train(args)
    t.start()