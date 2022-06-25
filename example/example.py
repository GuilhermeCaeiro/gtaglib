from gtaglib.generator import TagGenerator

dataset = [
    "One, two, three steps ahead.",
    "Next month, we will have to consider stepping up production to cover the high demand",
    "The second step of the plan is to meet the demand from local customers."
]

tag_generator = TagGenerator(
    semantic_field_size=40, 
    stemmer = "porter", 
    generate_bigrams=True,
    use_tfidf=True
)
abstract_tags, set_summary_tags, differential_tags = tag_generator.generate(dataset, method=1, root="rights")

print("1 --- ", abstract_tags, set_summary_tags, differential_tags)

tag_generator = TagGenerator(
    semantic_field_size=40, 
    stemmer="porter", 
    generate_bigrams=True,
    use_tfidf=True
)

set_summary_tags, differential_tags = tag_generator.generate_tag_cloud(
    dataset, 
    1,
    root="step", 
    expand_doc_tags = True, 
    max_additions = 3,
    outputdir="method1"
)

print("2 --- ", set_summary_tags, differential_tags)

tag_generator = TagGenerator(
    semantic_field_size=40, 
    stemmer="porter", 
    generate_bigrams=True,
    use_tfidf=True
)

set_summary_tags, differential_tags = tag_generator.generate_tag_cloud(
    dataset, 
    2,
    expand_doc_tags = True, 
    max_additions = 3,
    outputdir="method2"
)

print("3 --- ", set_summary_tags, differential_tags)