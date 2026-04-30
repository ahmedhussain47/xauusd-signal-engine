import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-tiny', device_map='cpu', dtype=torch.float32)

# Test with dummy price series
prices = torch.tensor([100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 102.0, 105.0, 103.0] * 13)
print('Input shape:', prices.shape)

# Tokenize
context = pipeline.tokenizer.context_input_transform(prices.unsqueeze(0))
print('Tokenized OK')

# Pass through encoder
input_ids = context[0] if isinstance(context, tuple) else context
with torch.no_grad():
    encoder_out = pipeline.model.model.encoder(input_ids=input_ids)
    hidden = encoder_out.last_hidden_state
    print('Hidden state shape:', hidden.shape)
    embedding = hidden.mean(dim=1)
    print('Embedding shape:', embedding.shape)
    print('Embedding extracted OK!')