# 🚀 Быстрый старт после исправления кластеризации

## ⚡ Немедленные действия

```bash
# 1. Очистить старый cache (ОБЯЗАТЕЛЬНО!)
./scripts/cleanup_and_test_clustering.sh

# 2. Запустить тестовый прогон
curl -X POST "http://localhost:8000/run/latest?sync=true"

# 3. Проверить результаты
ls -lt data/reports/*/  # найти последний отчет
cat data/reports/*/YOUR_VIDEO.cluster_diagnostics.json | python3 -m json.tool
```

## ✅ Что проверять

**Хорошие метрики:**
- `noise_ratio`: 0.15-0.30
- `silhouette`: 0.15-0.35  
- `assignment_confidence`: 0.85-0.95

**Если метрики плохие:**
1. Убедитесь что cache очищен
2. Проверьте что `.env` содержит новые настройки
3. Попробуйте более строгие пороги (см. CLUSTERING_FIXES_GUIDE.md)

## 📚 Полная документация

См. `CLUSTERING_FIXES_GUIDE.md` для детальной информации.
