# 🎯 Быстрое руководство: Исправленная кластеризация

## Что было исправлено

✅ **BGE модель** теперь получает правильные инструкции для embeddings  
✅ **Soft assignment** включен - комментарии правильно распределяются по кластерам  
✅ **PCA reduction** включена - убирает шум из больших embedding векторов  
✅ **Строгие quality gates** - только качественные кластеры проходят проверку  
✅ **Оптимальные пороги** - все параметры настроены для высокой точности  

## 🚀 Как запустить

### Шаг 1: Очистка старого cache (ОБЯЗАТЕЛЬНО!)
```bash
cd /media/herman/New\ Volume/YouTube_Projects/Github/YouTubeIntel
./scripts/cleanup_and_test_clustering.sh
```

Этот скрипт:
- Удалит старые embeddings без инструкций
- Очистит database cache
- Покажет текущие настройки

### Шаг 2: Запустить pipeline
```bash
# Через API (рекомендуется)
curl -X POST "http://localhost:8000/run/latest?sync=true"

# Или для конкретного видео
curl -X POST "http://localhost:8000/run/video?video_id=YOUR_VIDEO_ID&sync=true"
```

### Шаг 3: Проверить результаты
```bash
# Посмотреть последний отчет
ls -lt data/reports/ | head -5

# Проверить cluster diagnostics
cat data/reports/YYYY-MM-DD/VIDEO_ID.cluster_diagnostics.json | python3 -m json.tool | less
```

## 📊 Ожидаемые метрики

### ✅ ХОРОШИЕ метрики (цель):
```json
{
  "noise_ratio": 0.15-0.30,          // низкий процент шума
  "silhouette": 0.15-0.35,           // хорошее разделение
  "assignment_confidence": 0.85-0.95, // высокая уверенность
  "cluster_count": 5-8,               // разумное количество
  "emerging_cluster_count": 0-2,      // мало мусорных кластеров
  "ambiguous_share_pct": 10-18        // мало неоднозначных
}
```

### ❌ ПЛОХИЕ метрики (если видите - сообщите):
```json
{
  "noise_ratio": >0.40,              // слишком много шума
  "silhouette": <0.08,               // плохое разделение
  "assignment_confidence": <0.70,     // низкая уверенность
  "emerging_cluster_count": >3        // много мусорных кластеров
}
```

## 🔍 Быстрая проверка качества

Проверьте в сгенерированном отчете:
1. **Названия кластеров** - они должны точно описывать содержимое
2. **Подгруппы (positions)** - они должны быть связаны по смыслу
3. **Примеры комментариев** - в одном кластере должны быть похожие по теме
4. **Undetermined комментарии** - их должно быть <30%

## 🛠️ Тонкая настройка (если нужно)

Если результаты всё ещё не идеальны, можно дополнительно настроить в `.env`:

### Для ещё более строгих кластеров:
```env
CLUSTER_SOFT_ASSIGNMENT_MIN_SIMILARITY=0.55
TOPIC_COHERENCE_MIN=0.50
CLUSTER_ACCEPT_NOISE_RATIO=0.30
```

### Для меньшего количества подгрупп:
```env
CLUSTER_NOISE_SPLIT_MAX_GROUPS=2
POSITION_SUBCLUSTER_MAX_K=0
```

### Для более агрессивной редукции:
```env
CLUSTER_REDUCTION_TARGET_DIM=48
CLUSTER_REDUCTION_MIN_COMMENTS=100
```

## 🔄 Откат (если что-то пошло не так)

```bash
# Восстановить старые настройки
cp .env.backup .env

# Перезапустить сервис
# docker-compose restart app  # если используете Docker
```

## 📝 Важные файлы

- **Настройки**: `.env` (обновлён)
- **Backup настроек**: `.env.backup` (старые настройки)
- **Код embeddings**: `app/services/embeddings.py` (улучшен)
- **Код кластеризации**: `app/services/clustering.py` (оптимизирован)
- **Конфигурация**: `app/core/config.py` (настроена)
- **Cleanup скрипт**: `scripts/cleanup_and_test_clustering.sh`

## 🎓 Что изменилось технически

1. **BGE embeddings теперь с инструкциями**:
   ```
   Старый: "текст комментария"
   Новый: "Instruct: Represent the main topical subject...
           Text: текст комментария"
   ```

2. **Soft assignment работает**:
   - Каждый комментарий получает primary_confidence
   - Пограничные случаи размещаются в лучший кластер
   - Noise переназначается если similarity > 0.50

3. **Строгие качественные критерии**:
   - Принимается только noise_ratio < 0.35
   - Требуется silhouette > 0.08
   - Coherence должен быть > 0.45

## ❓ FAQ

**Q: Нужно ли удалять старые отчеты?**  
A: Нет, только cache embeddings (скрипт сделает это автоматически)

**Q: Сколько времени займёт первый запуск?**  
A: Немного дольше обычного - нужно сгенерировать новые embeddings

**Q: Можно ли вернуть E5 модель?**  
A: Да, просто измените `LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-large` в .env

**Q: Что делать если метрики всё ещё плохие?**  
A: Попробуйте более строгие настройки или вернитесь на E5 модель

## 🎉 Результат

После применения всех исправлений вы должны получить:
- ✅ Confidence 0.90-0.99 для кластеров (как было раньше!)
- ✅ Подгруппы семантически связаны
- ✅ Точные названия кластеров
- ✅ Минимум "мусорных" кластеров

---
**Создано**: $(date)  
**Версия исправлений**: v1.0
