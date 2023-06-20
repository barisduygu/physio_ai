# PhysioAI Chatbot

## Description

The PhysioAI Chatbot is a Streamlit application designed to extract information from multiple PDF documents. It utilizes OpenAI's language model for chat-like interactions. Users can upload PDF documents which the bot processes, and then ask questions regarding the uploaded documents. The bot then retrieves the information from the processed PDFs to answer the users' questions.

## Setup & Installation

1. Clone the repository:

```sh
git clone <repository-url>
```

2. Navigate into the project directory:

```sh
cd <project-dir>
```

3. Install the necessary libraries:

```sh
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```sh
streamlit run main.py
```

2. Open your web browser and go to `localhost:8501` to view the application.

3. Use the sidebar to upload a PDF document and click 'Process'.

4. After processing, you can ask questions related to the document. The chatbot will try to retrieve the information from the processed document to answer your question.

## Features

- OpenAI's language model for chat-like interactions.
- Extraction of information from multiple PDFs.
- Dynamic retrieval of information based on user queries.

## Directory Structure

- `VECTORSTORE_DIR`: This directory is used to store the vector representations of the processed PDF documents. These vectors are used to retrieve information during the chat interaction.
- `assets/logo`: This directory contains the logo image file used in the application.

## Limitations & Considerations

- The application requires a stable internet connection to function properly.
- The quality of the answers depends on the quality of the uploaded document and the relevance of the user's question to the document.
- Long or complex documents may take longer to process.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* Your Name

See also the list of [contributors](https://github.com/your-repo/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Acknowledge any contributors or sources that you have used in your project.