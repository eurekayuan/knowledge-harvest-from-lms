import json


out = open('pos_prompts.txt', 'w')
with open('relation_info/conceptnet.json') as f:
    relation_info = json.load(f)
for relation, info in relation_info.items():
    print(info['init_prompts'][0], file=out)
    for prompt in info['prompts']:
        print(prompt, file=out)

out.close()