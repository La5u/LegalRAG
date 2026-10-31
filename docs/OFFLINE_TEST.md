# 오프라인 검증 / Offline Test

문서가 외부로 나가지 않는다는 것을 말이 아니라 직접 확인하는 시험입니다. 누구나 같은
절차로 반복할 수 있습니다.

This test lets anyone confirm for themselves that LegalRAG works with no
internet connection.

## 절차 / Steps

1. [설치 안내](SETUP.md)의 2단계까지 마칩니다. / Finish step 2 of the
   [setup guide](SETUP.md), so the models and Neo4j image are downloaded.
2. **인터넷 연결을 끊습니다.** Wi-Fi를 끄고 랜선을 뽑으세요.
   / **Disconnect from the internet.** Turn off Wi-Fi and unplug Ethernet.
3. 검증을 실행합니다. / Run the check:

   ```bash
   ./scripts/offline-check
   ```

인터넷에 연결되어 있으면 스크립트는 시험을 거부합니다. 스크립트는 샘플 계약서
`documents/raw/samples/샘플_소프트웨어개발용역계약서.txt` 를 색인하고 다음을 질문합니다.

The script refuses to run while the internet is reachable. It indexes the
fictional sample contract and asks:

> 샘플 소프트웨어 개발 용역 계약서에서 납품이 늦어지면 지체상금은 하루에 얼마이고
> 최대 한도는 얼마인가요?

## 결과 확인 / Checking the answer

샘플 파일의 **제9조(지체상금)** 를 직접 열어 답변과 비교하세요.
/ Open **Article 9** of the sample file and compare it with the answer.

| 항목 / Item | 제9조의 내용 / In Article 9 |
|---|---|
| 하루당 / Per day | 계약금액의 1천분의 2.5 (제1항) |
| 한도 / Cap | 계약금액의 100분의 8 (제2항) |

이 값들은 일반적인 관행과 일부러 다르게 정했습니다. 모델이 일반 지식으로 맞힌 것이
아니라 계약서를 읽고 답했다는 근거가 됩니다.

The values are deliberately unusual, so a correct answer has to come from the
contract rather than from the model's general knowledge.

## 내 계약서로 시험하기 / Try your own contract

아직 오프라인 상태에서 [내 문서를 추가](SETUP.md#4-내-문서-추가--add-your-own-documents)하고
`./scripts/run` 으로 앱을 실행한 뒤, 답을 알고 있는 조항에 대해 질문해 보세요.

While still offline, add a contract of your own, start the app with
`./scripts/run`, and ask about a clause whose answer you already know.

## 이 시험이 증명하는 것 / What this shows

인터넷이 끊긴 상태에서 설치된 구성요소만으로 색인, 검색, 답변이 끝까지 동작한다는
것을 보여줍니다. 보안 감사를 대신하지는 않습니다.

It shows that indexing, retrieval, and answering work end to end with no
internet. It does not replace a security audit.
