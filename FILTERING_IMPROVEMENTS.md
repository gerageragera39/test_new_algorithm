# 🎯 Улучшение фильтрации комментариев

## Проблема
- emerging_other кластер был перемешанным (22% комментариев, coherence=0.65)
- Бессмысленные комментарии попадали в анализ
- Все комментарии в первом кластере были пограничными

## ✅ Что исправлено

### 1. Preprocessing: Новые фильтры
- 🚫 Эмоциональные реакции ("хахаха", "лол", "👍", "😂")
- 🚫 Generic похвала ("спасибо", "молодец", "топ", "❤")
- 🚫 Мета-комментарии ("первый", "алгоритм", "лайк если")
- 🚫 Низкокачественные вопросы ("когда видео?", "кто смотрит?")

### 2. Повышены пороги
- Минимум слов: 4 (было 3)
- Минимум букв: 8 (было 5)
- Lexical diversity: 0.42 (было 0.34)

### 3. Post-clustering фильтр
- Удаляет комментарии с confidence < 0.35
- Удаляет isolated комментарии (similarity < 0.25)

### 4. emerging_other фильтр
- Удаляет emerging_other если coherence < 0.70

## 📊 Ожидаемый результат

**До:**
- emerging_other: 22%, coherence=0.65 ❌
- Другие: coherence=0.76-0.80

**После:**
- Все кластеры: coherence > 0.75 ✅
- emerging_other: удалён или coherent ✅
- Меньше мусорных комментариев ✅

## 🚀 Тестирование

```bash
# Запустить pipeline
curl -X POST "http://localhost:8000/run/latest?sync=true"

# Проверить метрики
cat data/reports/*/VIDEO.cluster_diagnostics.json | jq '.clusters[] | {key: .cluster_key, coherence: .coherence_score}'
```

**Что проверять:**
- ✅ Все coherence > 0.75
- ✅ emerging_other отсутствует или coherence > 0.70
- ✅ Меньше комментариев в кластеризации
- ✅ Выше assignment_confidence

## 📁 Измененные файлы
- app/services/preprocessing.py - новые фильтры
- app/services/clustering.py - post-clustering filter
- app/core/config.py - новые пороги
- .env - настройки

Подробности: see `files/filtering-improvements-summary.md`
