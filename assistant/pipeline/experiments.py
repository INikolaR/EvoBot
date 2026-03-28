import json
import re
import sys
from typing import List, Tuple
from rag_service import RAGService

def load_dataset(path: str = "dataset.json") -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл {path} не найден.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Ошибка: Неверный формат JSON в файле {path}.")
        sys.exit(1)

def parse_binary_answer(text: str) -> int:
    if not text:
        return -1

    text_lower = text.lower()
    
    negative_markers = ["нет", "нельзя", "неверно", "запрещено", "не стоит", "не рекомендуется", "no", "false"]
    positive_markers = ["да", "можно", "верно", "разрешено", "yes", "true", "правильно"]

    neg_pos = len(text_lower)
    pos_pos = len(text_lower)

    for marker in negative_markers:
        idx = text_lower.find(marker)
        if idx != -1 and idx < neg_pos:
            neg_pos = idx

    for marker in positive_markers:
        idx = text_lower.find(marker)
        if idx != -1 and idx < pos_pos:
            pos_pos = idx

    if neg_pos < pos_pos:
        return 0
    elif pos_pos < len(text_lower):
        return 1
    else:
        return -1

def evaluate_rag(dataset: List[dict], service: RAGService) -> Tuple[float, List[dict]]:
    total = len(dataset)
    correct = 0
    results = []

    print(f"Начало оценки ({total} вопросов)...")

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"] # 1 или 0
        
        try:
            answer_text, _ = service.get_response(question)
            
            predicted_label = parse_binary_answer(answer_text)
            
            if predicted_label == -1:
                is_correct = False
                status = "AMBIGUOUS"
            else:
                is_correct = (predicted_label == ground_truth)
                status = "OK" if is_correct else "FAIL"
                if is_correct:
                    correct += 1
            
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "answer_text": answer_text,
                "status": status
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"Обработано: {i + 1}/{total}")

        except Exception as e:
            print(f"Ошибка при обработке вопроса {i}: {e}")
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_label": -1,
                "answer_text": "ERROR",
                "status": "ERROR"
            })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results

def print_report(accuracy: float, results: List[dict]):
    print(f"Всего вопросов: {len(results)}")
    print(f"Верных ответов: {sum(1 for r in results if r['status'] == 'OK')}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    dataset = load_dataset("data/else/closed_question_dataset.json")
    
    rag_service = RAGService()

    accuracy, results = evaluate_rag(dataset, rag_service)

    print_report(accuracy, results)
