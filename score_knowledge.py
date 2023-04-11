import json


with open('relation.json') as f:
    relation_info = json.load(f)

relations = []
for relation, info in relation_info.items():
    relations.append(relation)

with open('pos_prompts.txt') as f:
    prompts_pos = f.readlines()
with open('neg_prompts.txt') as f:
    prompts_neg = f.readlines()

def search_index(prompt, prompts):
    for i in range(len(prompts)):
        if prompts[i].strip() == prompt:
            return i
    return None


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
all_prompts_pos = []
all_prompts_neg = []

for relation in relations:
    entities_dir = f'results/conceptnet/{relation}/ent_tuples.json'
    with open(entities_dir) as f:
        entities = json.load(f)
        for ent0, ent1, prompt in entities:
            idx = search_index(prompt, prompts_pos)
            clip_prompt_pos = prompt.replace('<ENT0>', 'the object').replace('<ENT1>', ent1)
            clip_prompt_neg = prompts_neg[idx].replace('<ENT0>', 'the object').replace('<ENT1>', ent1).strip()
            
            all_prompts_pos.append((ent1, relation, clip_prompt_pos))
            all_prompts_neg.append((ent1, relation, clip_prompt_neg))


print(len(all_prompts_pos), len(all_prompts_neg))
