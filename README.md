# DevDox AI Agent

## Overview

The DevDox AI Agent is the core automation component of the DevDox AI platform, serving as the intelligent engine that interacts with Git repositories and code bases. This component enables developers to automate code creation, maintenance, and documentation, helping teams ship faster without burnout.

## Purpose and Functionality

The DevDox AI Agent is designed to:

- Automate code creation and maintenance based on user requirements
- Interact with Git repositories through APIs (GitHub, GitLab, etc.)
- Generate and manage pull/merge requests
- Create comprehensive code documentation
- Analyze code quality and suggest improvements
- Generate reports on code contributors, commits, and other Git metrics
- Integrate with CI/CD and deployment tools
- Process context from the DevDox AI Context service

## Technology Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Dependencies**:
  - PyGithub (GitHub API integration)
  - python-gitlab (GitLab API integration)
  - LangChain/LlamaIndex (Context processing)
  - Pydantic (Data validation)
  - SQLAlchemy (Database ORM)
  - Supabase-py (Database connectivity)
  - Redis (Optional - for caching and queue processing)

## Installation and Setup

### Prerequisites

- Python 3.10+
- Poetry (recommended for dependency management)
- Supabase account and credentials
- Git platform API credentials (GitHub, GitLab, etc.)

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/montymobile1/devdox-ai-agent.git
cd devdox-ai-agent

# Install dependencies using Poetry
poetry install

# Or using pip
pip install .

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Start the development server
poetry run uvicorn app.main:app --reload
```

## API Documentation

When running the application, the API documentation is automatically available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

The API provides endpoints for:

- Git repository operations
- Code generation and analysis
- Documentation generation
- Code contributor analytics
- Integration with CI/CD systems

## Configuration

Configuration is primarily managed through environment variables:

- `DEVDOX_CONTEXT_URL`: URL of the DevDox AI Context service
- `SUPABASE_URL`: Supabase instance URL
- `SUPABASE_KEY`: Supabase API key
- `GITHUB_TOKEN`: GitHub API token (if using GitHub integration)
- `GITLAB_TOKEN`: GitLab API token (if using GitLab integration)
- `LOG_LEVEL`: Logging level (default: INFO)

Additional configuration options can be found in `config.py`.

## Interaction with Other Components

The DevDox AI Agent interacts with:

1. **DevDox AI Context**: Consumes context information about code repositories to make intelligent decisions
2. **DevDox AI Portal API**: Receives requests from the portal through API calls
3. **Git platforms**: Connects to GitHub, GitLab, etc. via their APIs
4. **CI/CD systems**: Optionally integrates with deployment pipelines

## Development Guidelines

### Project Structure

```
devdox-ai-agent/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core functionality
│   ├── models/        # Data models
│   ├── services/      # Business logic
│   ├── utils/         # Utility functions
│   └── main.py        # Application entry point
├── tests/             # Test cases
├── .env.example       # Example environment variables
├── pyproject.toml     # Poetry dependencies
└── README.md          # This file
```

### Testing

Run tests using:

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=app
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

[MIT License](LICENSE)

---

*Related Jira Issue: [DV-1](https://montyholding.atlassian.net/browse/DV-1)*
