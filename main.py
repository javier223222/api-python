# app.py - API Flask para monitoreo de sistema solar fotovoltaico (sin caché)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
import os
import logging
import mysql.connector
from mysql.connector import Error
from flask import Flask, jsonify, request, send_file
import base64
from io import BytesIO

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

# Inicializar Flask
app = Flask(__name__)

# Configuración de BD
DB_CONFIG = {
    'host': '100.25.207.142',
    'port': '3306',
    'user': 'carlos',
    'password': 'carlos12345',
    'database': 'bd'
}

# ================== CLASE PARA PROCESAMIENTO DE DATOS ======================

class DataCleaner:
    """
    Clase para cargar, limpiar y procesar datos del sistema solar fotovoltaico
    """
    def __init__(self):
        self.electrical_data = None
        self.environment_data = None
        self.irradiance_data = None
        self.performance_metrics = {
            'processing_time': {},
            'data_quality': {}
        }
    
    def load_data_from_db(self, db_config):
        start_time = datetime.now()
    
        try:
            # Conectar a la base de datos
            logging.info(f"Conectando a la base de datos en {db_config['host']}...")
            conn = mysql.connector.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
        
            if conn.is_connected():
                logging.info("Conexión exitosa a la base de datos")
                cursor = conn.cursor(dictionary=True)
            
                # 1. Cargar datos eléctricos desde tabla1
                logging.info("Cargando datos de inversores desde tabla1...")
                query_electrical = "SELECT * FROM tabla1"
                cursor.execute(query_electrical)
                records_electrical = cursor.fetchall()
            
                if not records_electrical:
                    logging.error("No se encontraron registros en tabla1")
                    return False
                
                self.electrical_data = pd.DataFrame(records_electrical)
                logging.info(f"Cargados {len(self.electrical_data)} registros de datos eléctricos")
            
                # 2. Cargar datos de ambiente desde tabla2
                logging.info("Cargando datos ambientales desde tabla2...")
                query_environment = "SELECT * FROM tabla2"
                cursor.execute(query_environment)
                records_environment = cursor.fetchall()
            
                if records_environment:
                    self.environment_data = pd.DataFrame(records_environment)
                    logging.info(f"Cargados {len(self.environment_data)} registros de datos ambientales")
                else:
                    logging.warning("No se encontraron registros en tabla2. Usando datos simulados.")
                    self.environment_data = pd.DataFrame({
                        'measured_on': self.electrical_data['measured_on'],
                        'ambient_temperature_o_149575': 25.0,
                        'wind_speed_o_149576': 2.5,
                        'wind_direction_o_149577': 180.0
                    })
            
                # 3. Cargar datos de irradiancia desde tabla3
                logging.info("Cargando datos de irradiancia desde tabla3...")
                query_irradiance = "SELECT * FROM tabla3"
                cursor.execute(query_irradiance)
                records_irradiance = cursor.fetchall()
            
                if records_irradiance:
                    self.irradiance_data = pd.DataFrame(records_irradiance)
                    logging.info(f"Cargados {len(self.irradiance_data)} registros de datos de irradiancia")
                else:
                    logging.warning("No se encontraron registros en tabla3. Usando datos simulados.")
                    self.irradiance_data = pd.DataFrame({
                        'measured_on': self.electrical_data['measured_on'],
                        'poa_irradiance_o_149574': 800.0
                    })
            
                # Cerrar recursos
                cursor.close()
                conn.close()
                logging.info("Conexión cerrada")
            
                # Registrar métricas
                self.performance_metrics['processing_time']['loading'] = (datetime.now() - start_time).total_seconds()
                self.performance_metrics['data_quality']['records_loaded'] = {
                    'electrical': len(self.electrical_data),
                    'environment': len(self.environment_data),
                    'irradiance': len(self.irradiance_data)
                }
            
                return True
            else:
                logging.error("No se pudo conectar a la base de datos")
                return False
            
        except Error as e:
            logging.error(f"Error al conectar a MySQL: {e}")
            return False
        except Exception as e:
            logging.error(f"Error general al cargar datos: {str(e)}")
            return False
    
    def clean_electrical_data(self):
        """
        Limpia y prepara los datos eléctricos para su análisis
        """
        if self.electrical_data is None:
            logging.error("No hay datos eléctricos para limpiar")
            return False
        
        start_time = datetime.now()
        
        try:
            # Convertir 'measured_on' a datetime
            self.electrical_data['measured_on'] = pd.to_datetime(self.electrical_data['measured_on'])
            
            # Ordenar por fecha
            self.electrical_data = self.electrical_data.sort_values('measured_on')
            
            # Convertir todas las columnas numéricas a float
            for col in self.electrical_data.columns:
                if col != 'measured_on':
                    self.electrical_data[col] = pd.to_numeric(self.electrical_data[col], errors='coerce')
            
            # También limpiar los datos de ambiente e irradiancia
            if self.environment_data is not None:
                self.environment_data['measured_on'] = pd.to_datetime(self.environment_data['measured_on'])
                for col in self.environment_data.columns:
                    if col != 'measured_on':
                        self.environment_data[col] = pd.to_numeric(self.environment_data[col], errors='coerce')
                
                # Ordenar por fecha
                self.environment_data = self.environment_data.sort_values('measured_on')
            
            if self.irradiance_data is not None:
                self.irradiance_data['measured_on'] = pd.to_datetime(self.irradiance_data['measured_on'])
                for col in self.irradiance_data.columns:
                    if col != 'measured_on':
                        self.irradiance_data[col] = pd.to_numeric(self.irradiance_data[col], errors='coerce')
                
                # Ordenar por fecha
                self.irradiance_data = self.irradiance_data.sort_values('measured_on')
            
            # Registrar métricas
            self.performance_metrics['processing_time']['cleaning'] = (datetime.now() - start_time).total_seconds()
            
            return True
            
        except Exception as e:
            logging.error(f"Error al limpiar datos eléctricos: {str(e)}")
            return False
    
    def validate_data_quality(self):
        """
        Evalúa la calidad de los datos y retorna métricas
        """
        metrics = {
            'processing_time': self.performance_metrics['processing_time'],
            'completeness': {},
            'missing_values': {}
        }
        
        if self.electrical_data is not None:
            # Calcular porcentaje de valores completos
            total_cells = self.electrical_data.size
            missing_cells = self.electrical_data.isnull().sum().sum()
            completeness = (1 - missing_cells / total_cells) * 100
            
            metrics['completeness']['electrical'] = round(completeness, 2)
            metrics['missing_values']['electrical'] = int(missing_cells)
        
        return metrics

# ================== FUNCIONES PARA VISUALIZACIÓN ======================

def plot_power_chart(electrical_data):
    """
    Genera el gráfico de "Potencia AC por Inversor" y devuelve la imagen en base64
    """
    # Selecciona las columnas de potencia AC y asegura que son numéricas
    ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
    if not ac_power_cols:
        logging.error("No se encontraron columnas de potencia AC para graficar.")
        return None
    
    # Convertir a numérico y limpiar NaN
    data_to_plot = electrical_data.copy()
    for col in ac_power_cols:
        data_to_plot[col] = pd.to_numeric(data_to_plot[col], errors='coerce')
    
    # Filtrar solo inversores que tienen datos válidos
    active_inverters = []
    for col in ac_power_cols:
        if data_to_plot[col].notna().any() and data_to_plot[col].max() > 0:
            active_inverters.append(col)
    
    if not active_inverters:
        logging.error("No hay inversores con datos válidos para graficar.")
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Limitar al último día de datos válidos
    last_date = data_to_plot['measured_on'].max()
    start_date = last_date - pd.Timedelta(days=1)
    mask = (data_to_plot['measured_on'] >= start_date)
    data_to_plot = data_to_plot[mask]
    
    # Graficar cada inversor activo
    for col in active_inverters:
        inv_num = col.split('_')[1]  # Extraer número de inversor
        plt.plot(data_to_plot['measured_on'], 
                data_to_plot[col], 
                label=f'Inversor {inv_num}',
                linewidth=1)
    
    plt.title("Potencia AC por Inversor (kW)")
    plt.xlabel("Tiempo")
    plt.ylabel("Potencia AC (kW)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar en memoria
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convertir a base64 para web
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_environment_chart(environment_data):
    """
    Genera el gráfico de "Condiciones Ambientales" y devuelve la imagen en base64
    """
    # Verificar columnas necesarias
    temp_cols = [col for col in environment_data.columns if 'ambient_temperature' in col]
    wind_cols = [col for col in environment_data.columns if 'wind_speed' in col]
    
    if not temp_cols or not wind_cols:
        logging.error("No se encontraron las columnas necesarias para el gráfico de ambiente.")
        return None
    
    temp_col = temp_cols[0]  # Usar la primera columna que coincide
    wind_col = wind_cols[0]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Límite al último día (48 puntos para datos de 30 min)
    last_48_records = min(48, len(environment_data))
    data_to_plot = environment_data.iloc[-last_48_records:]
    
    # Temperatura Ambiente en eje izquierdo
    ax1.plot(data_to_plot['measured_on'], data_to_plot[temp_col],
             color='red', marker='o', label='Temperatura Ambiente (°C)')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Temperatura (°C)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    # Velocidad del Viento en eje derecho
    ax2 = ax1.twinx()
    ax2.plot(data_to_plot['measured_on'], data_to_plot[wind_col],
             color='blue', marker='s', label='Velocidad del Viento (m/s)')
    ax2.set_ylabel('Velocidad del Viento (m/s)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title("Condiciones Ambientales")
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    # Guardar en memoria
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convertir a base64 para web
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_correlation_chart(electrical_data, irradiance_data):
    """
    Genera un gráfico de correlación y devuelve la imagen en base64
    """
    try:
        # Preparar datos y asegurar que son numéricos
        electrical_data['measured_on'] = pd.to_datetime(electrical_data['measured_on'])
        irradiance_data['measured_on'] = pd.to_datetime(irradiance_data['measured_on'])
        
        # Calcular potencia total AC
        ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
        electrical_data['total_ac_power'] = pd.to_numeric(
            electrical_data[ac_power_cols].sum(axis=1), 
            errors='coerce'
        )
        
        # Buscar columna de irradiancia
        irradiance_cols = [col for col in irradiance_data.columns if 'poa_irradiance' in col or 'irradiance' in col]
        if not irradiance_cols:
            logging.error("No se encontraron columnas de irradiancia.")
            return None
        
        irradiance_col = irradiance_cols[0]
        irradiance_data[irradiance_col] = pd.to_numeric(irradiance_data[irradiance_col], errors='coerce')
        
        # Filtrar solo datos diurnos (donde hay irradiancia > 0)
        irradiance_data = irradiance_data[irradiance_data[irradiance_col] > 0]
        
        # Unir datasets
        merged_data = pd.merge_asof(
            electrical_data.sort_values('measured_on')[['measured_on', 'total_ac_power']],
            irradiance_data.sort_values('measured_on')[['measured_on', irradiance_col]],
            on='measured_on',
            direction='nearest',
            tolerance=pd.Timedelta('5min')
        )
        
        # Filtrar datos válidos y durante el día
        merged_data = merged_data.dropna()
        merged_data = merged_data[
            (merged_data['total_ac_power'] > 0) & 
            (merged_data[irradiance_col] > 0)
        ]
        
        # Crear figura con dos ejes Y
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Eje izquierdo: Irradiancia
        color1 = '#FFC107'  # Color amarillo para irradiancia
        ax1.set_xlabel('Hora')
        ax1.set_ylabel('Irradiancia (W/m²)', color=color1)
        line1 = ax1.plot(merged_data['measured_on'], 
                        merged_data[irradiance_col],
                        color=color1, 
                        label='Irradiancia Solar (W/m²)',
                        linewidth=2,
                        marker='o',
                        markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Eje derecho: Potencia AC
        ax2 = ax1.twinx()
        color2 = '#4CAF50'  # Color verde para potencia
        ax2.set_ylabel('Potencia AC (kW)', color=color2)
        line2 = ax2.plot(merged_data['measured_on'], 
                        merged_data['total_ac_power'],
                        color=color2, 
                        label='Producción Total AC (kW)',
                        linewidth=2,
                        marker='s',
                        markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Formatear eje X para mostrar solo horas
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()  # Rotar etiquetas
        
        # Añadir título y leyenda
        plt.title('Correlación Irradiancia - Potencia', pad=20)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center left', 
                  bbox_to_anchor=(1.15, 0.5))
        
        # Ajustar límites de los ejes para datos válidos
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar en memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        
        # Convertir a base64 para web
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
        
    except Exception as e:
        logging.error(f"Error al generar gráfico de correlación: {str(e)}")
        return None

# ================== FUNCIONES PARA ANÁLISIS DE DATOS ======================

def identify_inverter_columns(electrical_data):
    """
    Identifica y agrupa todas las columnas de inversores en el dataset
    """
    inverter_columns = {}
    
    # Buscar todos los inversores disponibles en el conjunto de datos
    for col in electrical_data.columns:
        if 'inv_' in col:
            try:
                # Extraer el número de inversor 
                if '_inv_' in col:
                    # Formato: inv_01_ac_power_inv_149583
                    inv_id = col.split('_inv_')[0]  # 'inv_01_ac_power'
                    inv_num = inv_id.split('_')[1]  # '01'
                else:
                    # Buscar otro formato posible
                    parts = col.split('_')
                    for i, part in enumerate(parts):
                        if part == 'inv' and i > 0:
                            inv_num = parts[i-1]
                            break
                    else:
                        continue  # Si no se encontró el número, saltar al siguiente
                
                # Inicializar el diccionario para este inversor si no existe
                if inv_num not in inverter_columns:
                    inverter_columns[inv_num] = {'dc_voltage': None, 'dc_current': None, 'ac_power': None}
                
                # Clasificar esta columna según su tipo
                if 'dc_voltage' in col:
                    inverter_columns[inv_num]['dc_voltage'] = col
                elif 'dc_current' in col:
                    inverter_columns[inv_num]['dc_current'] = col
                elif 'ac_power' in col:
                    inverter_columns[inv_num]['ac_power'] = col
                
            except Exception as e:
                logging.error(f"Error al procesar columna {col}: {e}")
    
    return inverter_columns

def compute_inverter_stats(electrical_data, inverter_columns):
    """
    Calcula estadísticas para cada inversor a partir de las columnas identificadas.
    """
    latest_data = electrical_data.iloc[-1]  # Último registro
    inverter_stats = []
    
    for inv_num, columns in inverter_columns.items():
        stats = {'inverter': f'inv_{inv_num}', 'status': 'Inactivo'}
        
        # Extraer valores actuales
        dc_voltage = 0
        dc_current = 0
        ac_power = 0
        
        if columns['dc_voltage']:
            try:
                dc_voltage = float(latest_data[columns['dc_voltage']])
                stats['dc_voltage'] = dc_voltage
            except (ValueError, TypeError):
                stats['dc_voltage'] = 0
        
        if columns['dc_current']:
            try:
                dc_current = float(latest_data[columns['dc_current']])
                stats['dc_current'] = dc_current
            except (ValueError, TypeError):
                stats['dc_current'] = 0
        
        if columns['ac_power']:
            try:
                ac_power = float(latest_data[columns['ac_power']])
                stats['ac_power'] = ac_power
            except (ValueError, TypeError):
                stats['ac_power'] = 0
        
        # Calcular eficiencia solo si hay datos válidos
        if dc_voltage > 0 and dc_current > 0:
            dc_power = dc_voltage * dc_current  # Potencia DC en W
            ac_power_w = ac_power * 1000  # Convertir kW a W
            efficiency = min(100, (ac_power_w / dc_power) * 100) if dc_power > 0 else 0
            stats['efficiency'] = round(efficiency, 1)
        else:
            stats['efficiency'] = 0
        
        # Definir el estado basado en los valores
        if ac_power > 0:
            stats['status'] = 'Activo'
        elif dc_voltage > 0 or dc_current > 0:
            stats['status'] = 'Advertencia'  # Hay DC pero no produce AC
        
        # Añadir estadísticas adicionales
        if columns['ac_power']:
            # Calcular promedio de potencia diaria (últimas 48 muestras)
            last_day_data = electrical_data.iloc[-48:][columns['ac_power']]
            stats['daily_average'] = float(last_day_data.mean())
            
            # Calcular máxima potencia del día
            stats['daily_max'] = float(last_day_data.max())
            
        inverter_stats.append(stats)
    
    # Ordenar por número de inversor
    inverter_stats.sort(key=lambda x: x['inverter'])
    
    return inverter_stats

def compute_dashboard_metrics(electrical_data, environment_data, irradiance_data):
    """
    Calcula las métricas principales para el dashboard con sus tendencias
    """
    # Preparar datos
    ac_power_cols = [col for col in electrical_data.columns if 'ac_power' in col]
    
    # Calcular potencia total si no existe
    if 'total_ac_power' not in electrical_data.columns and ac_power_cols:
        electrical_data['total_ac_power'] = electrical_data[ac_power_cols].sum(axis=1)
    
    # Buscar columnas de ambiente e irradiancia
    temp_cols = [col for col in environment_data.columns if 'ambient_temperature' in col]
    wind_cols = [col for col in environment_data.columns if 'wind_speed' in col]
    irradiance_cols = [col for col in irradiance_data.columns if 'poa_irradiance' in col or 'irradiance' in col]
    
    # Obtener datos más recientes
    latest_electrical = electrical_data.iloc[-1]
    latest_environment = environment_data.iloc[-1]
    latest_irradiance = irradiance_data.iloc[-1]
    
    # Obtener datos de 24 horas antes (o primer registro si no hay suficientes datos)
    prev_idx_electrical = max(0, len(electrical_data) - 49)  # -48 puntos = -24 horas
    prev_idx_environment = max(0, len(environment_data) - 49)
    prev_idx_irradiance = max(0, len(irradiance_data) - 49)
    
    prev_electrical = electrical_data.iloc[prev_idx_electrical]
    prev_environment = environment_data.iloc[prev_idx_environment]
    prev_irradiance = irradiance_data.iloc[prev_idx_irradiance]
    
    # Inicializar métricas
    metrics = {}
    
    # Métrica: Potencia Total AC
    if 'total_ac_power' in electrical_data.columns:
        total_ac_power = float(latest_electrical['total_ac_power'])
        prev_ac_power = float(prev_electrical['total_ac_power'])
        
        diff_ac = total_ac_power - prev_ac_power
        pct_ac = (diff_ac / prev_ac_power * 100) if prev_ac_power > 0 else 0
        
        metrics['total_ac_power'] = {
            'value': round(total_ac_power, 1),
            'unit': 'kW',
            'diff': round(diff_ac, 1),
            'diff_pct': round(pct_ac, 1),
            'trend': 'up' if diff_ac > 0 else 'down'
        }
    
    # Métrica: Irradiancia POA
    if irradiance_cols:
        irradiance_col = irradiance_cols[0]
        irradiance = float(latest_irradiance[irradiance_col])
        prev_irradiance_val = float(prev_irradiance[irradiance_col])
        
        diff_irr = irradiance - prev_irradiance_val
        pct_irr = (diff_irr / prev_irradiance_val * 100) if prev_irradiance_val > 0 else 0
        
        metrics['irradiance'] = {
            'value': round(irradiance, 1),
            'unit': 'W/m²',
            'diff': round(diff_irr, 1),
            'diff_pct': round(pct_irr, 1),
            'trend': 'up' if diff_irr > 0 else 'down'
        }
    
    # Métrica: Temperatura Ambiente
    if temp_cols:
        temp_col = temp_cols[0]
        temp = float(latest_environment[temp_col])
        prev_temp = float(prev_environment[temp_col])
        
        diff_temp = temp - prev_temp
        
        metrics['temperature'] = {
            'value': round(temp, 1),
            'unit': '°C',
            'diff': round(diff_temp, 1),
            'trend': 'up' if diff_temp > 0 else 'down'
        }
    
    # Métrica: Velocidad del Viento
    if wind_cols:
        wind_col = wind_cols[0]
        wind = float(latest_environment[wind_col])
        prev_wind = float(prev_environment[wind_col])
        
        diff_wind = wind - prev_wind
        
        metrics['wind_speed'] = {
            'value': round(wind, 1),
            'unit': 'm/s',
            'diff': round(diff_wind, 1),
            'trend': 'up' if diff_wind > 0 else 'down'
        }
    
    return metrics

def generate_alerts(electrical_data, environment_data, inverter_stats):
    """
    Genera alertas basadas en condiciones y métricas actuales
    """
    alerts = []
    
    # Parámetros para alertas (umbrales)
    low_power_threshold = 5.0  # kW
    high_temp_threshold = 40.0  # °C
    
    # Alertas por baja potencia en inversores
    for inv in inverter_stats:
        if inv['status'] == 'Activo' and inv.get('ac_power', 0) < low_power_threshold:
            alerts.append({
                'type': 'warning',
                'message': f"{inv['inverter'].replace('_', ' ').upper()} - Baja potencia detectada",
                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'icon': '⚠'
            })
        elif inv['status'] == 'Advertencia':
            alerts.append({
                'type': 'critical',
                'message': f"{inv['inverter'].replace('_', ' ').upper()} - Caída de potencia detectada",
                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'icon': '!'
            })
    
    # Alerta por temperatura
    temp_cols = [col for col in environment_data.columns if 'ambient_temperature' in col]
    if temp_cols:
        latest_temp = float(environment_data.iloc[-1][temp_cols[0]])
        if latest_temp > high_temp_threshold:
            alerts.append({
                'type': 'warning',
                'message': f"Temperatura elevada - {round(latest_temp, 1)}°C",
                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'icon': '⚠'
            })
    
    # Añadir alerta informativa (mantenimiento)
    alerts.append({
        'type': 'info',
        'message': "Mantenimiento preventivo programado - Bloque 2",
        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'icon': 'i'
    })
    
    return alerts

def generate_dashboard_data(cleaner):
    """
    Función principal que genera todos los datos del dashboard
    """
    try:
        # Validar que los datos estén cargados
        if cleaner.electrical_data is None or len(cleaner.electrical_data) == 0:
            logging.error("No hay datos eléctricos disponibles")
            return None
        
        # Asegurar que measured_on es datetime y ordenar los datos
        for df in [cleaner.electrical_data, cleaner.environment_data, cleaner.irradiance_data]:
            df['measured_on'] = pd.to_datetime(df['measured_on'])
            df.sort_values('measured_on', inplace=True)
        
        # 1. Generar gráficos en base64
        charts = {
            'power_chart': plot_power_chart(cleaner.electrical_data),
            'environment_chart': plot_environment_chart(cleaner.environment_data),
            'correlation_chart': plot_correlation_chart(
                cleaner.electrical_data, cleaner.irradiance_data)
        }
        
        # 2. Análisis de inversores
        inverter_columns = identify_inverter_columns(cleaner.electrical_data)
        inverter_stats = compute_inverter_stats(cleaner.electrical_data, inverter_columns)
        
        # 3. Métricas del dashboard
        metrics = compute_dashboard_metrics(
            cleaner.electrical_data,
            cleaner.environment_data,
            cleaner.irradiance_data
        )
        
        # 4. Generar alertas
        alerts = generate_alerts(
            cleaner.electrical_data,
            cleaner.environment_data,
            inverter_stats
        )
        
        # Crear estructura de datos del dashboard
        dashboard_data = {
            'charts': charts,
            'inverter_stats': inverter_stats,
            'metrics': metrics,
            'alerts': alerts,
            'last_update': datetime.now().isoformat()
        }
        
        return dashboard_data
    
    except Exception as e:
        logging.error(f"Error al generar datos del dashboard: {str(e)}")
        return None

# ================== RUTAS DE LA API ======================

def load_fresh_data():
    """
    Carga datos frescos desde la base de datos para cada petición
    """
    try:
        # Inicializar limpiador
        cleaner = DataCleaner()
        
        # Cargar datos desde la BD
        if not cleaner.load_data_from_db(DB_CONFIG):
            logging.error("No se pudieron cargar datos desde la base de datos")
            return None
        
        # Limpiar datos
        cleaner.clean_electrical_data()
        
        return cleaner
    except Exception as e:
        logging.error(f"Error al cargar datos frescos: {str(e)}")
        return None

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Endpoint para obtener datos del dashboard"""
    try:
        # Cargar datos frescos para cada petición
        cleaner = load_fresh_data()
        if not cleaner:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 500
        
        # Generar datos del dashboard
        dashboard_data = generate_dashboard_data(cleaner)
        if not dashboard_data:
            return jsonify({'error': 'Error al generar datos del dashboard'}), 500
        
        # Devolver solo datos, sin las imágenes en base64 (se acceden por separado)
        data_without_charts = dashboard_data.copy()
        if 'charts' in data_without_charts:
            del data_without_charts['charts']
        
        return jsonify(data_without_charts)
    
    except Exception as e:
        logging.error(f"Error en endpoint /api/dashboard: {str(e)}")
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/charts/<chart_name>', methods=['GET'])
def get_chart(chart_name):
    """Endpoint para obtener gráficos específicos"""
    try:
        # Cargar datos frescos para cada petición
        cleaner = load_fresh_data()
        if not cleaner:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 500
        
        # Generar los gráficos según el nombre solicitado
        if chart_name == 'power_chart':
            chart_data = plot_power_chart(cleaner.electrical_data)
        elif chart_name == 'environment_chart':
            chart_data = plot_environment_chart(cleaner.environment_data)
        elif chart_name == 'correlation_chart':
            chart_data = plot_correlation_chart(cleaner.electrical_data, cleaner.irradiance_data)
        else:
            return jsonify({'error': f'Gráfico {chart_name} no soportado'}), 404
        
        if not chart_data:
            return jsonify({'error': 'El gráfico no se pudo generar'}), 500
        
        # Devolver imagen como Base64
        return jsonify({'chart': chart_data})
    
    except Exception as e:
        logging.error(f"Error en endpoint /api/charts/{chart_name}: {str(e)}")
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Endpoint para obtener solo métricas"""
    try:
        # Cargar datos frescos para cada petición
        cleaner = load_fresh_data()
        if not cleaner:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 500
        
        # Calcular métricas
        metrics = compute_dashboard_metrics(
            cleaner.electrical_data,
            cleaner.environment_data,
            cleaner.irradiance_data
        )
        
        return jsonify(metrics)
    
    except Exception as e:
        logging.error(f"Error en endpoint /api/metrics: {str(e)}")
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/inverters', methods=['GET'])
def get_inverters():
    """Endpoint para obtener estadísticas de inversores"""
    try:
        # Cargar datos frescos para cada petición
        cleaner = load_fresh_data()
        if not cleaner:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 500
        
        # Analizar inversores
        inverter_columns = identify_inverter_columns(cleaner.electrical_data)
        inverter_stats = compute_inverter_stats(cleaner.electrical_data, inverter_columns)
        
        return jsonify(inverter_stats)
    
    except Exception as e:
        logging.error(f"Error en endpoint /api/inverters: {str(e)}")
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Endpoint para obtener alertas"""
    try:
        # Cargar datos frescos para cada petición
        cleaner = load_fresh_data()
        if not cleaner:
            return jsonify({'error': 'No se pudieron cargar los datos'}), 500
        
        # Analizar inversores para generar alertas
        inverter_columns = identify_inverter_columns(cleaner.electrical_data)
        inverter_stats = compute_inverter_stats(cleaner.electrical_data, inverter_columns)
        
        # Generar alertas
        alerts = generate_alerts(
            cleaner.electrical_data,
            cleaner.environment_data,
            inverter_stats
        )
        
        return jsonify(alerts)
    
    except Exception as e:
        logging.error(f"Error en endpoint /api/alerts: {str(e)}")
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/')
def index():
    """Página principal que muestra documentación básica de la API"""
    return """
    <h1>API de Monitoreo Solar Fotovoltaico</h1>
    <p>Endpoints disponibles:</p>
    <ul>
        <li><code>/api/dashboard</code> - Datos completos del dashboard</li>
        <li><code>/api/charts/power_chart</code> - Gráfico de potencia</li>
        <li><code>/api/charts/environment_chart</code> - Gráfico de condiciones ambientales</li>
        <li><code>/api/charts/correlation_chart</code> - Gráfico de correlación potencia-irradiancia</li>
        <li><code>/api/metrics</code> - Métricas principales del sistema</li>
        <li><code>/api/inverters</code> - Estado y estadísticas de inversores</li>
        <li><code>/api/alerts</code> - Alertas del sistema</li>
    </ul>
    <p><strong>Nota:</strong> Todos los endpoints se conectan directamente a la base de datos para obtener los datos más actualizados.</p>
    """

# ================== INICIALIZACIÓN DE LA APLICACIÓN ======================

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)