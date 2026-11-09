let chartInstance = null;

async function loadDropdownsAndMetrics() {
  try {
    const storeRes = await fetch('http://127.0.0.1:5000/api/stores');
    if (storeRes.ok) {
      const storeData = await storeRes.json();
      const storeSelect = document.getElementById('storeSelect');
      const deptSelect = document.getElementById('deptSelect');
      
      storeSelect.innerHTML = '';
      deptSelect.innerHTML = '';
      
      storeData.stores.forEach(s => storeSelect.add(new Option(`Store ${s}`, s)));
      storeData.departments.forEach(d => deptSelect.add(new Option(`Dept ${d}`, d)));
    }
  } catch (err) {
    console.warn('Backend API offline or data missing, populating default IDs.');
    const storeSelect = document.getElementById('storeSelect');
    const deptSelect = document.getElementById('deptSelect');
    for (let i = 1; i <= 45; i++) storeSelect.add(new Option(`Store ${i}`, i));
    for (let d = 1; d <= 99; d++) deptSelect.add(new Option(`Dept ${d}`, d));
  }

  try {
    const metricsRes = await fetch('http://127.0.0.1:5000/api/metrics');
    const metrics = await metricsRes.json();
    populateMetricsTable(metrics);
  } catch (err) {
    console.error('Error fetching metrics:', err);
  }

  fetchForecast();
}

function populateMetricsTable(metrics) {
  const tbody = document.querySelector('#metricsTable tbody');
  tbody.innerHTML = '';
  
  let bestWMAE = Infinity;
  let bestModelKey = '';
  
  Object.keys(metrics).forEach(k => {
    if (metrics[k].WMAE < bestWMAE) {
      bestWMAE = metrics[k].WMAE;
      bestModelKey = k;
    }
  });

  Object.keys(metrics).forEach(model => {
    const row = document.createElement('tr');
    if (model === bestModelKey) row.classList.add('best-model');
    row.innerHTML = `<td>${model} ${model === bestModelKey ? '🏆 (Best Model)' : ''}</td>
                     <td>$${metrics[model].MAE.toLocaleString()}</td>
                     <td>$${metrics[model].RMSE.toLocaleString()}</td>
                     <td>$${metrics[model].WMAE.toLocaleString()}</td>`;
    tbody.appendChild(row);
  });
}

async function fetchForecast() {
  const store_id = document.getElementById('storeSelect').value || 1;
  const dept_id = document.getElementById('deptSelect').value || 1;
  const model_name = document.getElementById('modelSelect').value;

  try {
    const res = await fetch('http://127.0.0.1:5000/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ store_id, dept_id, model_name })
    });
    
    const data = await res.json();
    document.getElementById('recommendationText').innerText = data.recommendation;

    renderChart(data.dates, data.actual, data.predicted);
  } catch (err) {
    document.getElementById('recommendationText').innerText = 'Make sure Flask backend is running on http://127.0.0.1:5000';
  }
}

function renderChart(labels, actual, predicted) {
  const ctx = document.getElementById('salesChart').getContext('2d');
  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Actual Weekly Sales ($)',
          data: actual,
          borderColor: '#0071ce',
          backgroundColor: 'rgba(0, 113, 206, 0.1)',
          fill: true,
          tension: 0.2
        },
        {
          label: 'Model Forecast ($)',
          data: predicted,
          borderColor: '#e03a3e',
          borderDash: [5, 5],
          fill: false,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        y: {
          title: { display: true, text: 'Sales ($)' },
          beginAtZero: false
        },
        x: {
          title: { display: true, text: 'Week Date' }
        }
      }
    }
  });
}

window.onload = loadDropdownsAndMetrics;
