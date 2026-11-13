\# 🛡️ Sentinel Semantic Layer (`sentinel-semantic`)



\[!\[Sentinel Semantic CI](https://github.com/yurtseveronder-design/sentinel-semantic/actions/workflows/ci.yml/badge.svg)](https://github.com/yurtseveronder-design/sentinel-semantic/actions)

\[!\[Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green.svg)](https://fastapi.tiangolo.com/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



semantic routing microservice and metadata extraction gateway designed for seamless integration with \*\*`sentinel-ui`\*\* and \*\*`ai-sentinel`\*\*. 



`sentinel-semantic` bridges natural language user prompts with secure, context-aware backend workflows—combining vector embedding similarity, multi-intent detection, PII/security guardrail tagging, and dynamic RAG filter generation.



\---



\## 🏗️ Core Architecture \& Features



1\. \*\*Semantic Intent Router (`src/semantic/router.py`)\*\*

&#x20;  - \*\*Embedding Similarity:\*\* Leverages `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity to match user prompts against anchor sentences.

&#x20;  - \*\*Robust Fallback Engine:\*\* Features graceful keyword/heuristic fallback handling for restricted or isolated environments (such as sandboxed Windows/AppLocker policies).

&#x20;  - \*\*Multi-Intent Detection:\*\* Detects overlapping intents matching confidence thresholds ($\\ge 0.60$).

&#x20;  - \*\*Decoupled Configuration (`config/intents.yaml`):\*\* Externalizes intent anchors to support multi-language definitions (English and Turkish) without requiring code changes or redeployments.



2\. \*\*Metadata \& RAG Extractor (`src/semantic/metadata.py`)\*\*

&#x20;  - Translates natural language queries into structured vector database filters (department, year, access level) while sanitizing clean search strings for RAG retrieval.



3\. \*\*FastAPI REST Gateway (`src/api/main.py`)\*\*

&#x20;  - Equipped with \*\*CORS middleware\*\* for seamless cross-service UI communication, strict validation via Pydantic, and clean OpenAPI documentation out of the box.



4\. \*\*Security \& Guardrail Handshakes\*\*

&#x20;  - Automatically flags sensitive data exposure, PII patterns (emails/SSNs), and security bypass/jailbreak attempts to trigger aggressive shielding policies in `ai-sentinel`.



\---



\## 📂 Project Structure



```text

sentinel-semantic/

├── .github/

│   └── workflows/

│       └── ci.yml             # GitHub Actions automated pytest CI pipeline

├── config/

│   └── intents.yaml           # Externalized multi-language intent anchors \& thresholds

├── src/

│   ├── api/

│   │   └── main.py            # FastAPI gateway endpoints \& CORS config

│   └── semantic/

│       ├── router.py          # Semantic router \& security/PII tagger

│       └── metadata.py        # RAG metadata extraction engine

├── tests/

│   └── test\_router.py         # Comprehensive pytest verification suite

├── Dockerfile                 # Lightweight containerization spec

├── pyproject.toml             # Project metadata and dependencies

└── README.md

