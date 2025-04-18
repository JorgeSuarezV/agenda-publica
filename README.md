# NewsArticleGen

## Overview
NewsArticleGen is an AI-powered tool designed to scan web news and generate preliminary article ideas for professionals. The system analyzes current news, creates topic suggestions across various categories, and matches these ideas with suitable professional profiles, streamlining the content creation process.

## Features
- **News Analysis**: Scans and processes news articles from various sources
- **Idea Generation**: Creates article ideas across multiple categories including International, National, Economy, Science & Technology, Health, Environment, Crime & Justice, Education, and Social & Humanitarian topics
- **Professional Matching**: Uses embeddings to find the most suitable professionals for each topic
- **Article Generation**: Creates preliminary article drafts based on selected ideas and professional profiles
- **Multilingual Support**: Processes and generates content in Spanish (with potential for other languages)

## Project Structure
- `create_embeddings.py`: Generates embeddings for professional profiles
- `find_professional.py`: Matches topics with appropriate professionals
- `generate_ideas.py`: Creates article ideas from news content
- `generate_article.py`: Produces preliminary article drafts
- `python_dump.py`: Contains utility functions used throughout the project
- `myutils/embeddings_utils.py`: Provides embedding-related utility functions

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/newsarticlegen.git
cd newsarticlegen
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key as an environment variable
```bash
export OPENAI_API_KEY='your-api-key'
```

4. Create the necessary directories
```bash
mkdir -p resources/raw_profiles resources/embedded_profiles
```

## Usage

### 1. Prepare professional profiles
Add professional profile text files to the `resources/raw_profiles` directory.

### 2. Generate embeddings for profiles
```bash
python create_embeddings.py
```

### 3. Generate article ideas from news
- Add news content to `resources/News.txt`
- Run the idea generator:
```bash
python generate_ideas.py
```

### 4. Find professionals for specific topics
- Add your topic to `resources/User_prompt_find_professional.txt`
- Run:
```bash
python find_professional.py
```

### 5. Generate articles
- Set up your article request in `resources/User_prompt_generate_article.txt`
- Run:
```bash
python generate_article.py
```

## Customization
- Modify the prompts in the Python files to adjust the style and focus of generated content
- Adjust the number of ideas per category in `generate_ideas.py`
- Change the matching criteria in `find_professional.py`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Specify your license here]

## Acknowledgments
- Built using OpenAI's API
- Uses embedding techniques for semantic matching
