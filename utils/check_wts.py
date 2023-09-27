import torch
from nets.net_e2e import Feat_Merger
from nets.net_e2e_text import Feat_Merger as Feat_Merger_text

model1 = Feat_Merger()
model2 = Feat_Merger_text()
model1.load_state_dict(torch.load("/home/singh/CARE_India/care_india_arch/models/model-e2espeech-simLoss-B16-E10-lr1e-4-K10-nsample20.pt"))
model2.load_state_dict(torch.load("/home/singh/CARE_India/care_india_arch/models/model-e2etext_ESPAttnLastLayerRM-B16-E10-lr1e-4-K10-nsample20.pt"))

print(model1.self_attn_layer.linear_k.weight)
print(model2.self_attn_layer.linear_k.weight)

