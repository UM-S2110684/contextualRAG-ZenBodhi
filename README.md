# contextualRAG-ZenBodhi
# Buddhism Chatbot - Contextual RAG System

This project is a **Buddhism Chatbot** that leverages **Contextual RAG (Retrieval-Augmented Generation)** to provide accurate and contextually relevant answers to questions related to **Theravāda** and **Mahāyāna** Buddhist traditions. The chatbot is designed to differentiate between overlapping terminologies and principles in these two traditions, ensuring high accuracy and clarity in its responses.

---

## Features

- **Contextual Differentiation**: The chatbot accurately distinguishes between **Theravāda** and **Mahāyāna** content, even when terminologies overlap.
- **Accuracy Metrics**: The system evaluates answers based on **Content Inclusion**, **Contextual Alignment**, and **Clarity**.
- **Cost-Effective Token Usage**: The chatbot integrates **Deepseek-v3** for cost-effective and high-performance answer generation.
- **Comparison Analysis**: The chatbot can analyze and compare concepts from both traditions, providing detailed insights into their philosophical and practical differences.
- **Error Handling**: The system includes robust error handling to manage exceptions during document processing, retrieval, and answer generation.

---

## System Architecture

The system is built using the following components:

- **LangChain**: For document processing and retrieval.
- **ChromaDB**: For vector storage and semantic search.
- **HuggingFace**: For embedding and reranking models.
- **OpenRouter**: For deploying the chatbot and integrating models like **Claude-3.5-Sonnet** and **Deepseek-v3**.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/buddhism-chatbot.git
   cd buddhism-chatbot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file and add your **OpenRouter API Key**:
     ```bash
     OPENROUTER_API_KEY=your_api_key_here
     ```

4. **Run the Chatbot**:
   ```bash
   python buddhism_chatbot_v1.py
   ```

---

## Usage

### Chatbot Interaction
- Start the chatbot by running the script:
  ```bash
  python buddhism_chatbot_v1.py
  ```
- Type your questions, and the chatbot will provide detailed, contextually accurate answers.
- To exit the chatbot, type `exit`.

### Evaluating Answers
- To evaluate the accuracy of answers, use the `process_answers.py` script:
  ```bash
  python process_answers.py input.csv
  ```
  This will generate a CSV file with evaluation results.

---

## Evaluation Metrics

The system uses the following metrics to evaluate the quality of generated answers:

1. **Content Inclusion**: Measures how well the answer captures key points from the standard answer.
2. **Contextual Alignment**: Evaluates how well the answer maintains the context and meaning of the standard answer.
3. **Clarity**: Assesses the clarity and coherence of the answer using readability metrics and linguistic analysis.

---

## Results

The **Contextual RAG System** achieves **80% accuracy** in contextual differentiation, outperforming other models like **ZenBodhi** and **QAnything Rerank**. The system is only **5% less accurate** than the current state (**Norbu**), demonstrating its capability to provide high-quality, contextually aligned answers.

---

## Future Work

- **Improve Error Handling**: Add more robust error handling for edge cases.
- **Expand Knowledge Base**: Include more Buddhist texts and traditions to enhance the chatbot's knowledge.
- **Optimize Token Usage**: Further reduce token costs while maintaining high performance.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

---

## Acknowledgments

- **OpenRouter** for providing the API for model integration.
- **HuggingFace** for the embedding and reranking models.
- **LangChain** for document processing and retrieval.

---