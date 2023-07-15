import fairseq
import torch
import torchaudio
from fairseq.models.hubert import HubertModel

ckpt_path = r"C:\Users\autumn\Downloads/checkpoint_best_legacy_500.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path],'s')
model = models[0].to('cuda').eval()
# model.skip_masked=True
# model.skip_nomask =True
waveform, _ = torchaudio.load('./test1/2099003695.wav')
emissions, _ = model.extract_features(waveform.to('cuda'),padding_mask = torch.BoolTensor(waveform.shape).fill_(False).to('cuda'))
emissions = emissions[0].cpu().detach().transpose(1, 0)
emissions=emissions.unsqueeze(0)

pass