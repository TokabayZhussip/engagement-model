# Данные

Сами данные в репозиторий не коммитятся (см. `.gitignore`). Скачайте их и положите по путям ниже.

## 1. OULAD — основной датасет

Open University Learning Analytics Dataset: 7 курсов, 22 потока (2013–2014), около 32 тыс. студентов, около 10 млн записей о кликах. Лицензия CC-BY 4.0.

- Официальная страница: https://analyse.kmi.open.ac.uk/open_dataset
- Зеркало UCI: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

Распакуйте архив так, чтобы получилось:

```
data/raw/oulad/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv        (~450 МБ)
└── vle.csv
```

Ссылка для статьи: Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset // Scientific Data. 2017. Vol. 4. 170171. https://doi.org/10.1038/sdata.2017.171

## 2. Синтетические данные (для проверки кода)

```
python scripts/make_synthetic_oulad.py --out data/raw/synthetic
```

Генерирует маленький набор в формате OULAD. Нужен только для тестов и CI, в статью эти результаты не идут.

## 3. Расширения (в текущей версии кода не реализованы)

- **KDD Cup 2015 (XuetangX)** — для проверки переносимости индекса на другой MOOC. Нужен адаптер, который переводит логи событий (`problem`, `video`, `access`, `discussion` …) в недельные признаки того же формата, что в `engagement/features.py`.
- **DAiSEE / EngageNet** — эмоциональный компонент. Модель по видео выдаёт `emotion_score` ∈ [0, 1] на студента и неделю. Таблица `data/processed/emotion.csv` с колонками `code_module, code_presentation, id_student, week, emotion_score` подхватывается автоматически, если в `config.yaml` задан `index.emotional_weight > 0`.
