# prompt_builder_paper.py — Prompt engineering for paper extraction

SYSTEM_PROMPT = """You are a scientific metadata extraction specialist.
Your task is to extract two structured fields from a paper's metadata:
  • scientific_domains  — broad scientific/medical disciplines the paper belongs to
  • research_focuses    — specific research topics, study objects, or application areas the paper covers

Input fields available: Title, Abstract, Keywords. Use Keywords and Abstract as primary sources.

Extraction rules:
1. Use the Keywords field as the PRIMARY source for scientific_domains when available.
2. Use the Abstract text as the PRIMARY source for research_focuses AND as evidence for both fields.
3. For each domain/focus, capture 1–2 SHORT evidence phrases (≤8 words each) verbatim from the Abstract.
4. Labels must be noun phrases in English, title-cased (e.g. "Clinical Neurology").
5. scientific_domains: broad disciplines, 1–5 labels max.
6. research_focuses: specific topics or methodologies, 2–7 labels max.
7. Do NOT hallucinate domains not supported by text. Do NOT add comments inside JSON.
8. Evidence MUST be exact quoted substrings from the Abstract — never inferred or paraphrased.
9. Output ONLY a valid RFC-8259 JSON object. No markdown fences, no comments, no trailing commas, no explanation.

Output schema (strict):
{
  "scientific_domains": ["Domain1", "Domain2"],
  "scientific_domains_evidence": {
    "Domain1": ["exact phrase from abstract", "another exact phrase"],
    "Domain2": ["exact phrase from abstract"]
  },
  "research_focuses": ["Focus1", "Focus2", "Focus3"],
  "research_focuses_evidence": {
    "Focus1": ["exact phrase from abstract"],
    "Focus2": ["exact phrase from abstract"]
  }
}"""

FEW_SHOT_EXAMPLES = [
    {
        "title": "Deep Learning for Medical Image Segmentation",
        "keywords": "Deep Learning; Medical Imaging; Segmentation",
        "abstract": (
            "We propose a convolutional neural network architecture for the automated segmentation "
            "of medical images. Experiments on MRI and CT datasets demonstrate accurate lesion "
            "segmentation and improvements over classical methods. The method targets clinical "
            "applications in radiology and diagnostic imaging."
        ),
        "output": """{
  "scientific_domains": ["Medical Imaging", "Machine Learning"],
  "scientific_domains_evidence": {
    "Medical Imaging": ["medical images", "clinical applications in radiology"],
    "Machine Learning": ["convolutional neural network", "deep learning"]
  },
  "research_focuses": [
    "Automated Medical Image Segmentation",
    "CNN Architectures for Segmentation",
    "Lesion Detection and Segmentation"
  ],
  "research_focuses_evidence": {
    "Automated Medical Image Segmentation": ["automated segmentation of medical images"],
    "CNN Architectures for Segmentation": ["convolutional neural network architecture"],
    "Lesion Detection and Segmentation": ["lesion segmentation"]
  }
}"""
    }
]


def build_prompt_paper(title: str, keywords: str, abstract: str) -> list[dict]:
    few_shot_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_block += f"""
--- EXAMPLE {i} ---
Title: {ex['title']}
Keywords: {ex['keywords']}
Abstract: {ex['abstract']}
Output:
{ex['output']}
"""

    user_content = f"""Below are {len(FEW_SHOT_EXAMPLES)} worked examples showing the expected extraction format:
{few_shot_block}
--- YOUR TASK ---
Now extract for this paper. Follow the same format exactly.

Title: {title}
Keywords: {keywords if keywords else 'Not provided'}
Abstract: {abstract if abstract else 'Not provided'}

Output:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_prompt_paper_no_system(title: str, keywords: str, abstract: str) -> list[dict]:
    few_shot_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_block += f"""
--- EXAMPLE {i} ---
Title: {ex['title']}
Keywords: {ex['keywords']}
Abstract: {ex['abstract']}
Output:
{ex['output']}
"""

    user_content = f"""{SYSTEM_PROMPT}

Below are {len(FEW_SHOT_EXAMPLES)} worked examples:
{few_shot_block}
--- YOUR TASK ---
Title: {title}
Keywords: {keywords if keywords else 'Not provided'}
Abstract: {abstract if abstract else 'Not provided'}

Output:"""

    return [{"role": "user", "content": user_content}]
