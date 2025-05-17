## Proyecto final de moduli

## Introducción / Objetivo del Proyecto

## ideas

- imaginemos una extensión de browser que use un modelo ML de phishig, qué nos interesa? evitar errores tipo 1 o tipo 2? qué es más doloroso para usuarios? evidentemente
  podríamos hacer lana mostrando ads cada vez que bloqueamos una phishing URL y penalizar cuando NO sea phishing.
  duele mas error tipo 2, la metrica sería recall.

- el dataset es muy simple porque la parte difícil del análisis léxico estático de las URLs ya está hecho, por lo tanto es más importante darle una interpretación a las predicciones del modelo y tratar de entender porqué clasifica de cierta manera.

- tiene explicaciones basadas en la psicología más que en técnicas avanzadas de compu, los humanos "escanean" la url rápido y si parece legítima bbva bbvaa hscb o con prefijos parecidos entonces son engañados a navegar hacia ella. los atacantes usan tecnicas de enmascarar dominios malos como buenos, con prefijos o con técnicas de encoding en la url, redirecciones o trucos no reconocibles a simple vista.

## Colaboradores

- **Fernando Alvarado Palacios**: [GitHub](https://github.com/Fernando-Alvarado) [Linkedin](https://www.linkedin.com/in/fernando-alvarado-37415b216/)

## 📈 Métodos Utilizados

## 🔧 Tecnologías

- R
- Python
- Rmarkdown

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![RMarkdown](https://img.shields.io/badge/RMarkdown-2C3E50?style=flat&logo=r&logoColor=white)

## 📊 Descripción del Proyecto

## 💡 Necesidades del Proyecto

- Estadísticas descriptivas
- Limpieza y procesamiento de datos
- Modelado estadístico
- Redacción y documentación

## Cómo Empezar

Descarga Proyecto

```bash
git clone https://github.com/Fernando-Alvarado/PhishML.git
```

## ▶️ Ejecución del Proyecto

## 📁 Organizacion del proyecto

## ⚙️ Requisitos

```text

PHISHML/
├── 📂Data/                   # Datos originales y procesados
│   ├── 📂Raw/               # Datos originales (sin modificar)
│   └── 📂Procesada/         # Datos limpios y transformados
├── 📂Notebooks/             # Exploraciones o pruebas interactivas
│   ├── 📂Python/            # Notebooks en Python
│   └── 📂R/                 # Notebooks en R
├── 📂Outputs/               # Resultados generados
│   ├── 📂graficas/          # Figuras, gráficos, visualizaciones
│   └── 📂resultados/        # Tablas, métricas, predicciones
├── 📂Reports/               # Reportes del proyecto
│   └── 🅡reporteFinal.Rmd   # Código fuente del informe final
├── 📂Scripts/               # Código modular
│   ├── 📂Python/            # Scripts de modelado en Python
│   └── 📂R/                 # Scripts de modelado en R
├── .gitignore             # Archivos y carpetas ignoradas por Git
├── LICENSE                # Licencia del proyecto
└── README.md              # Descripción general del proyecto




```

## ✅ Organización del trabajo con GitHub Projects

Para llevar un control claro de nuestras tareas, usamos un **tablero tipo Kanban** en GitHub Projects. Este está dividido en columnas que indican el estado de cada actividad.

### 🟢 Backlog

- Tareas identificadas que **aún no se han empezado**.
- Puede haber ideas, tareas pendientes sin responsable o cosas a largo plazo.

### 🔵 Ready

- Tareas **listas para empezar a trabajar**.
- Ya están definidas y se pueden tomar directamente desde aquí.

### 🟡 In progress

- Tareas que **ya están en proceso** por alguien del equipo.
- Solo deberías tener 1 o 2 tareas en esta columna a la vez.

### 🟣 In review

- Tareas que **ya terminaron**, pero que necesitan ser **revisadas** antes de marcarse como completas.

### 🟠 Done

- Tareas **completadas y revisadas**.
- No necesitan más atención.

---

### 📌 ¿Cómo usar el tablero?

1. Crea una tarjeta con tu tarea usando `+ Add item`.
2. Mueve la tarjeta entre columnas según tu avance.
3. Asigna responsable y etiquetas desde el _Issue_ vinculado.
4. En cada reunión o revisión, usamos este tablero para ver avances y bloqueos.

---

🔗 **Accede al tablero aquí:**  
[Ir al tablero de tareas](https://github.com/users/Fernando-Alvarado/projects/1/views/1)
