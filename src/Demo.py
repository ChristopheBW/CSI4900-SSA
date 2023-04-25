from paddlenlp.transformers import SkepForTokenClassification, SkepForSequenceClassification, SkepTokenizer, SkepModel
import paddle
from seqeval.metrics.sequence_labeling import get_entities

label_ext_path = "./data/opener_en_extraction/label.dict"
label_cls_path = "./data/opener_en_relations/label.dict"
ext_model_path = "./checkpoint/best_ext.pdparams"
cls_model_path = "./checkpoint/best_cls.pdparams"


def load_ext_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


def load_cls_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


model_name = "skep_ernie_2.0_large_en"
ext_label2id, ext_id2label = load_ext_dict(label_ext_path)
cls_label2id, cls_id2label = load_cls_dict(label_cls_path)
tokenizer = SkepTokenizer.from_pretrained(model_name)
print("label dict loaded.")

# load ext model
ext_state_dict = paddle.load(ext_model_path)
ext_skep = SkepModel.from_pretrained(model_name)
ext_model = SkepForTokenClassification(ext_skep, num_classes=len(ext_label2id))
ext_model.load_dict(ext_state_dict)
print("extraction model loaded.")

# load cls model
cls_state_dict = paddle.load(cls_model_path)
cls_skep = SkepModel.from_pretrained(model_name)
cls_model = SkepForSequenceClassification(cls_skep, num_classes=len(cls_label2id))
cls_model.load_dict(cls_state_dict)
print("classification model loaded.")


def decoding(text, tag_seq):
    words = text.split(" ")
    assert len(words) == len(tag_seq), f"text len: {len(text)}, tag_seq len: {len(tag_seq)}"
    puncs = list(",.?;!")
    splits = [idx for idx in range(len(words)) if words[idx] in puncs]
    print(f'words: {words}')

    prev = 0
    sub_texts, sub_tag_seqs = [], []
    for i, split in enumerate(splits):
        sub_tag_seqs.append(tag_seq[prev:split])
        sub_texts.append(words[prev:split])
        prev = split
    sub_tag_seqs.append(tag_seq[prev:])
    sub_texts.append((words[prev:]))
    # print(f"sub_tag_seqs: {sub_tag_seqs}")

    ents_list = []
    for sub_text, sub_tag_seq in zip(sub_texts, sub_tag_seqs):
        ents = get_entities(sub_tag_seq, suffix=False)
        ents_list.append((sub_text, ents))

    # print(f'ents_list: {ents_list}')
    aps = []
    no_a_words = []
    for sub_tag_seq, ent_list in ents_list:
        sub_aps = []
        sub_no_a_words = []
        # print(ent_list)
        for ent in ent_list:
            ent_name, start, end = ent
            if ent_name == "Aspect":
                aspect = sub_tag_seq[start:end + 1]
                sub_aps.append([aspect])

                if len(sub_no_a_words) > 0:
                    sub_aps[-1].extend(sub_no_a_words)
                    sub_no_a_words.clear()
            else:
                ent_name == "Opinion"
                opinion = sub_tag_seq[start:end + 1]
                opinion = " ".join(opinion)
                # print(f"opinion: {opinion}")
                if len(sub_aps) > 0:
                    sub_aps[-1].append(opinion)
                else:
                    sub_no_a_words.append(opinion)

        if sub_aps:
            aps.extend(sub_aps)
            if len(no_a_words) > 0:
                aps[-1].extend(no_a_words)
                no_a_words.clear()
        elif sub_no_a_words:
            if len(aps) > 0:
                aps[-1].extend(sub_no_a_words)
            else:
                no_a_words.extend(sub_no_a_words)

    if no_a_words:
        no_a_words.insert(0, "None")
        aps.append(no_a_words)

    return aps


def is_aspect_first(text, aspect, opinion_word):
    return text.find(aspect) <= text.find(opinion_word)


def concate_aspect_and_opinion(text, aspect, opinion_words):
    aspect_text = ""
    if aspect == "None":
        aspect = ""
    for opinion_word in opinion_words:
        if is_aspect_first(text, aspect, opinion_word):
            aspect_text += aspect + opinion_word + ", "
        else:
            aspect_text += opinion_word + aspect + ", "
    aspect_text = aspect_text[:-1]

    return aspect_text


def format_print(results):
    for result in results:
        aspect, opinions, sentiment = result["aspect"], result["opinions"], result["sentiment"]
        print(f"aspect: {aspect}, opinions: {opinions}, sentiment: {sentiment}")
    print()


def predict(input_text, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label, max_seq_len=512):
    ext_model.eval()
    cls_model.eval()

    # processing input text
    text_list = input_text.split(" ")
    n = len(text_list)
    encoded_inputs = tokenizer(text_list, is_split_into_words=True, max_seq_len=max_seq_len, )
    input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
    token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

    # extract aspect and opinion words
    logits = ext_model(input_ids, token_type_ids=token_type_ids)
    predictions = logits.argmax(axis=2).numpy()[0]
    tag_seq = [ext_id2label[idx] for idx in predictions][1:-1]
    print(f'tag_seq: {tag_seq}')
    aps = decoding(text, tag_seq)
    # print(f'aps: {aps}')

    # predict sentiment for aspect with cls_model
    results = []
    for ap in aps:
        aspect = ap[0]
        opinion_words = list(set(ap[1:]))
        aspect_text = concate_aspect_and_opinion(input_text, aspect, opinion_words)

        encoded_inputs = tokenizer(aspect_text, text_pair=input_text, max_seq_len=max_seq_len, return_length=True)
        input_ids = paddle.to_tensor([encoded_inputs["input_ids"]])
        token_type_ids = paddle.to_tensor([encoded_inputs["token_type_ids"]])

        logits = cls_model(input_ids, token_type_ids=token_type_ids)
        prediction = logits.argmax(axis=1).numpy()[0]

        result = {"aspect": aspect, "opinions": opinion_words, "sentiment": cls_id2label[prediction]}
        results.append(result)

    # print results
    format_print(results)


max_seq_len = 512

# ask for user input
text = "a"
while text != "":
    text = input("Please input your text: ")
    predict(text, ext_model, cls_model, tokenizer, ext_id2label, cls_id2label, max_seq_len=max_seq_len)
