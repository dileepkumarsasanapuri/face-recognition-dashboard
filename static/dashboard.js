// static/dashboard.js
async function fetchDashboardData() {
    const res = await fetch("/dashboard_data");
    const data = await res.json();

    // Update total
    document.getElementById("total-count").innerText = data.total;

    // Update chart
    const ctx = document.getElementById("barChart").getContext("2d");
    if (window.myChart) window.myChart.destroy();
    window.myChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.names,
            datasets: [{
                label: "Recognitions",
                data: data.counts,
                backgroundColor: "rgba(75,192,192,0.6)"
            }]
        }
    });

    // Update table
    const logTable = document.getElementById("recent-logs");
    logTable.innerHTML = "";
    for (const log of data.recent) {
        const row = `<tr><td>${log.timestamp}</td><td>${log.name}</td></tr>`;
        logTable.innerHTML += row;
    }
}

setInterval(fetchDashboardData, 3000); // Refresh every 3s
fetchDashboardData();
