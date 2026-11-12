# Datos de ejemplo y datos públicos

## Dataset local reproducible

No es necesario descargar datos para ejecutar las pruebas o probar el flujo.
El generador determinista crea series diarias por tienda y producto:

```bash
python -c "from data.sample_data import write_sample_data; write_sample_data('data/sample.csv')"
```

El CSV sintético se crea bajo demanda y no se versiona. Sirve para validar
interfaces y ejemplos, no para inferir resultados comerciales reales.

## Kaggle: Store Item Demand Forecasting Challenge

1. Crea una cuenta de Kaggle y acepta las reglas del concurso:
   <https://www.kaggle.com/competitions/demand-forecasting-kernels-only>.
2. Instala y configura la CLI oficial de Kaggle con un token propio. No guardes
   `kaggle.json` ni credenciales dentro del repositorio.
3. Descarga y descomprime el dataset en `data/raw/`:

   ```bash
   kaggle competitions download -c demand-forecasting-kernels-only -p data/raw
   python -m zipfile -e data/raw/demand-forecasting-kernels-only.zip data/raw
   ```

4. El archivo `train.csv` contiene `date`, `store`, `item` y `sales`, compatible
   con los módulos del proyecto. Mantén los datos descargados localmente: `data/raw`
   está excluido de Git.

Como alternativa, el dataset M5 está disponible en Kaggle:
<https://www.kaggle.com/competitions/m5-forecasting-accuracy>. Requiere adaptar
su formato ancho y las claves de jerarquía al esquema largo
`date, store, item, sales` antes de usar este motor.
