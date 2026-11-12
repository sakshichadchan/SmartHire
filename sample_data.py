"""
sample_data.py
--------------
Sample job descriptions (written for this project) and helpers to load the sample resume.
The job descriptions also serve as a small background corpus so that TF-IDF's IDF term
has more than two documents to learn from (see ml_model.py).
"""
from pathlib import Path

SAMPLE_RESUME_PATH = Path(__file__).parent / "sample_data" / "sample_resume.pdf"


def load_sample_resume_bytes() -> bytes:
    return SAMPLE_RESUME_PATH.read_bytes()


SAMPLE_JOBS: dict[str, str] = {
"Machine Learning Intern": """Machine Learning Intern - Data Science Team

About the role
We are looking for a motivated intern to help build and evaluate machine learning models on real customer data.

Requirements:
- Strong programming skills in Python
- Hands-on experience with Pandas, NumPy and Scikit-learn
- Understanding of machine learning algorithms, statistics and model evaluation
- Experience with SQL and relational databases
- Familiarity with data visualization using Matplotlib or Seaborn
- Knowledge of Git and GitHub
- Good communication and problem solving skills

Preferred qualifications:
- Experience with TensorFlow or PyTorch
- Exposure to NLP and deep learning
- Docker or cloud platforms such as AWS
- Experience building dashboards with Streamlit
""",
"Python Backend Developer": """Python Backend Developer

Responsibilities
Design, build and maintain scalable backend services and REST APIs for our web platform.

Requirements:
- Solid Python programming skills
- Experience with Django or Flask
- Building and consuming REST APIs
- Working knowledge of SQL and PostgreSQL
- Strong data structures and algorithms fundamentals
- Version control with Git
- Problem solving and teamwork

Nice to have:
- Docker and containerization
- AWS or other cloud services
- Redis caching
- CI/CD pipelines and unit testing
""",
"Full Stack Web Developer": """Full Stack Web Developer

We build modern web applications and need a developer comfortable across the stack.

Requirements:
- HTML, CSS and JavaScript
- React for building user interfaces
- Node.js and Express.js backend development
- MongoDB or another NoSQL database
- REST API design
- Git and GitHub
- Teamwork and communication

Preferred:
- TypeScript
- Docker
- AWS
- Experience working in Agile teams
""",
"Data Analyst": """Data Analyst

Turn business questions into insights using data.

Requirements:
- Advanced SQL for querying relational databases
- Microsoft Excel for analysis and reporting
- Python with Pandas for data analysis and data cleaning
- Data visualization using Power BI or Tableau
- Statistics fundamentals
- Strong communication and analytical skills

Nice to have:
- R programming
- Matplotlib and Seaborn
- Experience with A/B testing
- Attention to detail
""",
"Java Backend Developer": """Java Backend Developer

Build reliable enterprise services for our banking clients.

Requirements:
- Strong Java and object oriented programming
- Spring Boot and REST APIs
- MySQL or other relational databases
- Data structures and algorithms
- Microservices concepts
- Git

Good to have:
- Docker and Kubernetes
- AWS
- JUnit and unit testing
- Agile methodology
""",
"DevOps / Cloud Engineer (Entry Level)": """DevOps / Cloud Engineer (Entry Level)

Help us automate, deploy and monitor cloud infrastructure.

Requirements:
- Linux administration and Bash scripting
- Git version control
- Docker containerization
- AWS fundamentals
- CI/CD concepts and pipelines
- Computer networks basics
- Problem solving

Preferred qualifications:
- Kubernetes
- Terraform
- Python scripting
- Jenkins or GitHub Actions
- Microsoft Azure
""",
}
