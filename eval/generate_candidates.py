"""LLM-assisted eval sorusu üretimi.

data/index.db'deki her chunk için Foundry Local chat modeline "bu metne
dayanan bir soru-cevap çifti üret" diye sorar, çıktıyı eval/candidates_draft.json'a
yazar. Bu dosya OTOMATIK OLARAK dataset_tr.json'a eklenmez — bir insan
(SME) gözden geçirip iyi olanları elle taşımalıdır (bkz. dataset_tr.json'daki
"origin": "llm-assisted" girdileri).

Kullanım:
    .venv/Scripts/python.exe eval/generate_candidates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_rag.config import DEFAULT_DB_PATH  # noqa: E402
from local_rag.llm.foundry import FoundryChatProvider  # noqa: E402
from local_rag.storage import db  # noqa: E402

GENERATION_PROMPT = """\
Aşağıda kurumsal bir dokümandan alınmış bir metin parçası var. Bu metne
dayanarak, yalnızca bu metinden cevaplanabilecek TEK bir soru-cevap çifti
üret. Cevap kısa ve metne birebir dayalı olsun. Şu formatta yanıt ver, başka
hiçbir şey yazma:

SORU: <soru>
CEVAP: <cevap>
"""


def main() -> None:
    conn = db.get_connection(DEFAULT_DB_PATH)
    llm = FoundryChatProvider()

    rows = conn.execute(
        """
        SELECT c.text AS text, c.section_path AS section_path, d.filename AS filename,
               d.classification AS classification
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.chunk_type = 'text' AND c.injection_flag IS NULL AND d.classification = 'genel'
        ORDER BY d.filename, c.chunk_index
        """
    ).fetchall()

    candidates = []
    for row in rows:
        response = llm.chat(GENERATION_PROMPT, row["text"])
        question, _, answer = response.partition("CEVAP:")
        question = question.replace("SORU:", "").strip()
        answer = answer.strip()
        if not question or not answer:
            continue
        candidates.append(
            {
                "query": question,
                "expected_answer_hint": answer,
                "expected_source_filename": row["filename"],
                "section_path": row["section_path"],
                "origin": "llm-assisted",
            }
        )
        print(f"[{row['filename']}] {question} -> {answer}")

    out_path = Path(__file__).parent / "candidates_draft.json"
    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(candidates)} aday üretildi: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
