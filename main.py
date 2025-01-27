# def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
#     model.eval()
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
#     generated = input_ids
#
#     with torch.no_grad():
#         for _ in range(max_length):
#             logits = model(generated, mask=None)
#             next_token_logits = logits[:, -1, :] / temperature
#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
#             generated = torch.cat([generated, next_token], dim=-1)
#
#             # Oprire dacă se generează un token special (de ex., <end>)
#             if next_token.item() == tokenizer.eos_token_id:
#                 break
#
#     return tokenizer.decode(generated[0], skip_special_tokens=True)
#
# # Exemplu de utilizare
# prompt = "Once upon a time"
# generated_text = generate_text(model, tokenizer, prompt, max_length=50)
# print(generated_text)