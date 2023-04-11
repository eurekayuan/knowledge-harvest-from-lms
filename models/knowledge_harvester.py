from tqdm import tqdm
from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import fix_prompt_style, is_valid_prompt, get_n_ents, get_sent, chatgpt


class KnowledgeHarvester:
    def __init__(self,
                 model_name,
                 max_n_prompts=20,
                 max_n_ent_tuples=10000,
                 max_word_repeat=5,
                 max_ent_subwords=1,
                 prompt_temp=1.):
        self._weighted_prompts = []
        self._ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_word_repeat = max_word_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp

        self._model = None
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

        self._seed_ent_tuples = None

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    def set_prompts(self, prompts):
        for prompt in prompts:
            if is_valid_prompt(prompt=prompt):
                self._weighted_prompts.append([fix_prompt_style(prompt), 1.])

    def update_ent_tuples(self):
        ent_tuples = []
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for prompt, _ in self._weighted_prompts:
            assert get_n_ents(prompt) == 2
            for category in cifar10_classes:
                sent = get_sent(prompt=prompt, ent_tuple=(category, 'XXX'))
                raw_response = chatgpt(prompt=f'{sent}\nGive all the possible answers for XXX. The response should be words or phrases separated by commas. No additional sentences should be included.')
                attrs = raw_response.strip().strip('.').lower().split(', ')
                for attr in attrs:
                    ent_tuples.append((category, attr, prompt))
                    print(category, attr, prompt)
        self._ent_tuples = ent_tuples

    def score_ent_tuple(self, ent_tuple):
        score = 0.
        for prompt, weight in self.weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)

        return score

    def score(self, prompt, ent_tuple):
        logprobs = self._model.fill_ent_tuple_in_prompt(
            prompt=prompt, ent_tuple=ent_tuple)['mask_logprobs']

        token_wise_score = sum(logprobs) / len(logprobs)
        ent_wise_score = sum(logprobs) / len(ent_tuple)
        min_score = min(logprobs)

        return (token_wise_score + ent_wise_score + min_score) / 3.
    
    @property
    def ent_tuples(self):
        return self._ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts