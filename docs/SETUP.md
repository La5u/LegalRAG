# 설치 안내 / Setup Guide

LegalRAG는 내 컴퓨터에서만 동작합니다. 모델을 한 번 내려받은 뒤에는 인터넷 연결이
필요 없고, 문서가 외부로 전송되지 않습니다.

LegalRAG runs on your own machine. Once the models are downloaded it needs no
internet connection, and documents are not sent anywhere.

## 1. 준비물 / Requirements

- Linux, Python 3.11+
- Docker Compose 또는 Podman / Docker Compose or Podman
- [Ollama](https://ollama.com)

## 2. 설치 / Install (인터넷 필요 / internet required)

```bash
git clone https://github.com/La5u/LegalRAG.git
cd LegalRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 임베딩 및 답변 모델 내려받기 / download the embedding and answer models
HF_HOME=.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
ollama pull qwen3.5:4b

# Neo4j 이미지 내려받기 / pull the Neo4j image
docker compose pull neo4j
```

Neo4j 비밀번호를 기본값에서 바꾸세요. / Change the Neo4j password from the default:

```bash
export NEO4J_PASSWORD="$(openssl rand -hex 16)"
export LEGALRAG_NEO4J_PASSWORD="$NEO4J_PASSWORD"
```

처음 실행하기 전에 설정해야 합니다. 이 값을 안전한 곳에 보관하고 실행할 때마다 다시
설정하세요. / Set this before the first run, keep the value somewhere safe, and
set it again for each run.

## 3. 실행 / Run

```bash
./scripts/run
```

브라우저에서 <http://127.0.0.1:8000> 을 여세요. / Open <http://127.0.0.1:8000>.

서비스는 기본적으로 이 컴퓨터(127.0.0.1)에서만 접속할 수 있습니다. / Services
listen on 127.0.0.1 only by default.

## 4. 내 문서 추가 / Add your own documents

1. 파일(HWPX, 텍스트 PDF, TXT)을 `documents/raw/` 아래에 넣습니다.
   / Put the file under `documents/raw/`.
2. `documents/source_manifest.csv` 에 한 줄을 추가합니다.
   / Add a row to `documents/source_manifest.csv`:

   ```csv
   documents/raw/contracts/내계약서.txt,contract,Korea,ko,internal,,Confidential,true,
   ```

3. `./scripts/run` 을 다시 실행합니다. / Run `./scripts/run` again.

`documents/raw/` 는 git에서 제외되어 있으므로 내 문서가 저장소에 커밋되지 않습니다.
/ `documents/raw/` is git-ignored, so your files are not committed.

## 5. 설치 검증 / Verify

[오프라인 검증](OFFLINE_TEST.md)을 진행하세요. / Run the
[offline test](OFFLINE_TEST.md).

> 생성된 답변은 법률 자문이 아닙니다. 항상 인용된 원문을 확인하세요.
> Generated answers are not legal advice. Always check the cited text.
