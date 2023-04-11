from data_utils.data_utils import chatgpt


with open('pos_prompts.txt') as f:
    lines = f.readlines()


for line in lines:
    ret = chatgpt(f'Negative statement for: {line}\nNo additional sentences should be included.')
    with open('neg_prompts.txt', 'a') as f:
        print(ret, file=f)
