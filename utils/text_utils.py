import torch
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel


########## BERT #################
# Documentations for bert tokenizer: 
# https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertTokenizer

# Documentations for bert model:
# https://huggingface.co/docs/transformers/en/model_doc/distilbert?usage=AutoModel#transformers.DistilBertModel

# Available model for bert
# https://huggingface.co/distilbert


########## CLIP #################

# Documentation for clip tokenizer
# https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer

# Documentation for clip text model
# https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTextModel

# Available model for clip
# https://huggingface.co/openai


def get_tokenizer_and_model(model_type, device, eval_mode=True):
    assert model_type in ('bert', 'clip'), "Text model can only be one of clip or bert"
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    else:
        text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    if eval_mode:
        text_model.eval()
    return text_tokenizer, text_model
    

def get_text_representation(text, text_tokenizer, text_model, device,
                            truncation=True,
                            padding='max_length',
                            max_length=77):
    
    # Documentation on how to use pre-trained tokenizer, refer to the __call__ part
    # https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
    text_model.eval()
    with torch.no_grad():
    
        token_output = text_tokenizer(text,
                                      truncation=truncation, # Truncate to the max_length number of tokens
                                      padding=padding, # Pad to the  max_length
                                      return_attention_mask=True, # Return attention mask -> Mask that zero out the padding token in the attention map (similar to causal attention map)
                                      max_length=max_length)
        
        indexed_tokens = token_output['input_ids'] # Token in the form of numbers, its a list
        att_masks = token_output['attention_mask'] # What are attention_masks: https://huggingface.co/docs/transformers/v4.51.3/en/glossary#attention-mask
        
        # Conver the output into tensors
        tokens_tensor = torch.tensor(indexed_tokens).to(device)
        mask_tensor = torch.tensor(att_masks).to(device)
        
        # The structure of the model is the transformer (attention-based)
        # Therefore, think of .last_hidden_state as the output of the final attention block in the shape of B x L x context_dim
        # where L is the number of sequence length, likely to be 77 in this case given the max_length
        text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state 
        text_embed = text_embed.detach()
    
    return text_embed
