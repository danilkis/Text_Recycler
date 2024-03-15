from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc
)

# Инициализация компонентов натали
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

# Чтение файла
with open('ju.txt', 'r') as input_file:
    text = input_file.read()

# Обработка текста
doc = Doc(text)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)
doc.tag_ner(ner_tagger)

# Лемматизация
for token in doc.tokens:
    token.lemmatize(morph_vocab)

# Чтение из файла со стоп словами
with open('stopwords-ru.txt', 'r', encoding='utf-8') as stopwords_file:
    stop_words = set(stopwords_file.read().splitlines())

# Фильтрация стоп слов
filtered_tokens = [token for token in doc.tokens if token.lemma.lower() not in stop_words]

# Сохранение в финальный файл
with open('ju_processed.txt', 'w') as output_file:
    for token in filtered_tokens:
        output_file.write(f"{token.text} ({token.pos}, {token.lemma})\n")

print("Готово!")
