/**
 * Renderiza los gráficos del dashboard utilizando Chart.js con el tema de Shadcn/UI.
 * @param {Object} data - Objeto de datos inyectado desde el backend.
 */
function renderDashboard(data) {
    // Configuración global de Chart.js para consistencia con Shadcn/UI
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b'; // Slate-500
    Chart.defaults.scale.grid.color = '#f1f5f9'; // Slate-100
    Chart.defaults.plugins.tooltip.backgroundColor = '#0f172a'; // Slate-900
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: 600 };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: { displayColors: false }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { font: { size: 11 } }
            },
            y: {
                border: { display: false },
                grid: { borderDash: [4, 4] },
                ticks: {
                    callback: function(value) {
                        return 'S/ ' + value.toLocaleString();
                    },
                    font: { size: 11 }
                }
            }
        },
        elements: {
            bar: { borderRadius: 4 },
            line: { tension: 0.4 },
            point: { radius: 3, hitRadius: 10, hoverRadius: 6 }
        }
    };

    // Proyección de Ventas (Gráfico de Línea)
    if (data.proyeccion) {
        const ctxProyeccion = document.getElementById('chartProyeccion').getContext('2d');
        const labelsProyeccion = data.proyeccion.historico_fechas.concat(data.proyeccion.proyeccion_fechas);
        const valoresProyeccion = data.proyeccion.historico_valores.concat(data.proyeccion.proyeccion_valores);
        const historyLength = data.proyeccion.historico_valores.length;

        new Chart(ctxProyeccion, {
            type: 'line',
            data: {
                labels: labelsProyeccion,
                datasets: [{
                    label: 'Ventas',
                    data: valoresProyeccion,
                    borderColor: '#0ea5e9', // Sky-500
                    backgroundColor: 'transparent',
                    borderWidth: 2.5,
                    fill: false,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#0ea5e9',
                    pointBorderWidth: 2,
                    pointHoverBackgroundColor: '#0ea5e9',
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 2,
                    segment: {
                        borderDash: ctx => ctx.p0DataIndex >= historyLength - 1 ? [6, 6] : undefined,
                    }
                }]
            },
            options: {
                ...commonOptions,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    ...commonOptions.plugins,
                    tooltip: {
                        ...commonOptions.plugins.tooltip,
                        callbacks: {
                            title: (items) => {
                                const index = items[0].dataIndex;
                                if (index >= historyLength) return 'Proyección: ' + items[0].label;
                                return items[0].label;
                            }
                        }
                    }
                }
            }
        });
    }

    // Participación por Servicio (Gráfico de Dona)
    if (document.getElementById('chartPieServicio')) {
        new Chart(document.getElementById('chartPieServicio'), {
            type: 'doughnut',
            data: {
                labels: data.graficos_servicio.labels,
                datasets: [{
                    data: data.graficos_servicio.data,
                    backgroundColor: [
                        '#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#e0f2fe'
                    ],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'right',
                        labels: {
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { family: "'Inter', sans-serif", size: 12 },
                            padding: 20,
                            generateLabels: function(chart) {
                                const data = chart.data;
                                if (data.labels.length && data.datasets.length) {
                                    return data.labels.map(function(label, i) {
                                        const meta = chart.getDatasetMeta(0);
                                        const style = meta.controller.getStyle(i);
                                        const value = data.datasets[0].data[i];
                                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                                        const percentage = ((value / total) * 100).toFixed(1) + '%';
                                        
                                        return {
                                            text: `${label} (${percentage})`,
                                            fillStyle: style.backgroundColor,
                                            strokeStyle: style.borderColor,
                                            lineWidth: style.borderWidth,
                                            hidden: isNaN(data.datasets[0].data[i]) || meta.data[i].hidden,
                                            index: i
                                        };
                                    });
                                }
                                return [];
                            }
                        }
                    },
                    tooltip: {
                        ...commonOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1) + '%';
                                return `${context.label}: S/ ${value.toLocaleString()} (${percentage})`;
                            }
                        }
                    }
                },
                cutout: '65%',
                layout: { padding: 20 }
            }
        });
    }

    // Histograma de Pesos (Gráfico de Barras)
    if (document.getElementById('chartHistograma') && data.histograma) {
        new Chart(document.getElementById('chartHistograma'), {
            type: 'bar',
            data: {
                labels: data.histograma.bins,
                datasets: [{
                    label: 'Frecuencia',
                    data: data.histograma.counts,
                    backgroundColor: '#0ea5e9',
                    hoverBackgroundColor: '#0284c7',
                    borderRadius: 4,
                    barPercentage: 0.9,
                    categoryPercentage: 0.9
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 10 } }
                    },
                    y: {
                        border: { display: false },
                        grid: { color: '#f1f5f9' },
                        ticks: {
                            callback: function(value) {
                                return value; // Cantidad simple
                            },
                            font: { size: 11 }
                        }
                    }
                },
                plugins: {
                    ...commonOptions.plugins,
                    tooltip: {
                        ...commonOptions.plugins.tooltip,
                        callbacks: {
                            label: function (context) {
                                return context.parsed.y + ' envíos';
                            }
                        }
                    }
                }
            }
        });
    }

    // Facturación por Servicio (Gráfico de Barras)
    if (document.getElementById('chartBarServicio') && data.graficos_servicio) {
        new Chart(document.getElementById('chartBarServicio'), {
            type: 'bar',
            data: {
                labels: data.graficos_servicio.labels,
                datasets: [{
                    label: 'Total Facturado',
                    data: data.graficos_servicio.data,
                    backgroundColor: '#0ea5e9',
                    hoverBackgroundColor: '#0284c7',
                    borderRadius: 4,
                    barPercentage: 0.6,
                }]
            },
            options: commonOptions
        });
    }

    // Facturación por Destino (Gráfico de Barras)
    if (document.getElementById('chartBarDestino') && data.graficos_destino) {
        new Chart(document.getElementById('chartBarDestino'), {
            type: 'bar',
            data: {
                labels: data.graficos_destino.labels,
                datasets: [{
                    label: 'Total Facturado',
                    data: data.graficos_destino.data,
                    backgroundColor: '#38bdf8',
                    hoverBackgroundColor: '#0ea5e9',
                    borderRadius: 4,
                    barPercentage: 0.6,
                }]
            },
            options: commonOptions
        });
    }

    // Evolución de Peso Diario (Gráfico de Línea)
    if (document.getElementById('chartPesoDiario') && data.grafico_peso_diario) {
        new Chart(document.getElementById('chartPesoDiario'), {
            type: 'line',
            data: {
                labels: data.grafico_peso_diario.labels,
                datasets: [{
                    label: 'Peso (Kg)',
                    data: data.grafico_peso_diario.data,
                    borderColor: '#0ea5e9',
                    backgroundColor: 'rgba(14, 165, 233, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#0ea5e9'
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 11 } }
                    },
                    y: {
                        border: { display: false },
                        grid: { borderDash: [4, 4] },
                        ticks: {
                            callback: function(value) {
                                return value + ' Kg';
                            },
                            font: { size: 11 }
                        }
                    }
                },
                plugins: {
                    ...commonOptions.plugins,
                    tooltip: {
                        ...commonOptions.plugins.tooltip,
                        callbacks: {
                            label: function (context) {
                                return context.parsed.y + ' Kg';
                            }
                        }
                    }
                }
            }
        });
    }

    // Categorías de Envío (Gráfico de Pastel)
    if (document.getElementById('chartCategorias') && data.grafico_categorias) {
        new Chart(document.getElementById('chartCategorias'), {
            type: 'pie',
            data: {
                labels: data.grafico_categorias.labels,
                datasets: [{
                    data: data.grafico_categorias.data,
                    backgroundColor: [
                        '#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'right',
                        labels: {
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: { family: "'Inter', sans-serif", size: 11 }
                        }
                    }
                }
            }
        });
    }

    // Top Vendedores (Gráfico de Barras)
    if (document.getElementById('chartTopVendedores') && data.grafico_vendedores) {
        new Chart(document.getElementById('chartTopVendedores'), {
            type: 'bar',
            data: {
                labels: data.grafico_vendedores.labels,
                datasets: [{
                    label: 'Ventas',
                    data: data.grafico_vendedores.data,
                    backgroundColor: '#0ea5e9',
                    hoverBackgroundColor: '#0284c7',
                    borderRadius: 4,
                    barPercentage: 0.6,
                }]
            },
            options: commonOptions
        });
    }
}
