<div align="center" style="background-color:#2f3e4e; padding:32px 24px; border-radius:12px; margin-bottom:32px;">

<h1 style="color:white; margin-bottom:8px;">
Survival Analysis System
</h1>

<p style="color:#dbe3ea; font-size:16px; margin-top:0;">
Microservices architecture for competing risks analysis
</p>

</div>

### Описание
<hr style="border:none; height:2px; background-color:#3498DB; margin-bottom:24px;">

В области страхования и кредитного скоринга важную роль для поддержания финансовой устойчивости компаний играет анализ времени до наступления определенных событий, таких как досрочное погашение долга, дефолт или прекращение договора. Особенностью подобных задач является наличие сразу нескольких вероятных событий (и взаимоисключающих). Для решения подобных задач применяется анализ выживаемости с конкурирующими рисками. 

Данная система разработана в качестве технической платформы для воспроизводимой оценки качества и сравнения моделей анализа выживаемости в условиях конкурирующих рисков на наборе данных о кредитах на жилье, предоставляемом компанией Freddie Mac. Она предоставляет единый интерфейс для создания тестовых данных, применения их на различных моделей и расчёта метрик качества (IBS, AUPRC) с возможностью визуализации статистик и результатов.

### Инструкция по запуску
<hr style="border:none; height:2px; background-color:#3498DB; margin-bottom:24px;">

1. Копируем репозиторий

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Переходим в проект

```bash
cd ms-system/app
```

3. Собираем образы

```bash
docker compose build
```

4. Запускаем систему

```bash
docker compose up -d
start http://localhost:8000
```

