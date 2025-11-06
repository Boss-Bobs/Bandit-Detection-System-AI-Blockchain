// Configuration
const RPI_BASE_URL = 'https://relativistic-kacie-puffingly.ngrok-free.dev'; // Your ngrok URL
const INFURA_URL = 'https://sepolia.infura.io/v3/8742554fd5c94c549cb8b4117b076e7a';
const CONTRACT_ADDRESS = '0x279FcACc1eB244BBD7Be138D34F3f562Da179dd5';
const CONTRACT_ABI = [
    {"inputs": [{"internalType": "string", "name": "_folder", "type": "string"}, {"internalType": "uint256", "name": "_frameIdx", "type": "uint256"}, {"internalType": "string", "name": "_error", "type": "string"}], "name": "logAnomaly", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "stateMutability": "nonpayable", "type": "constructor"},
    {"inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "name": "anomalies", "outputs": [{"internalType": "string", "name": "folder", "type": "string"}, {"internalType": "uint256", "name": "frameIdx", "type": "uint256"}, {"internalType": "string", "name": "error", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}], "name": "getAnomaly", "outputs": [{"internalType": "string", "name": "", "type": "string"}, {"internalType": "uint256", "name": "", "type": "uint256"}, {"internalType": "string", "name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getAnomalyCount", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "owner", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"}
];
const POLL_INTERVAL = 30000; // 30 seconds

// Initialize Web3 for blockchain queries
let web3;
let contract;
async function initWeb3() {
    web3 = new Web3(INFURA_URL);
    contract = new web3.eth.Contract(CONTRACT_ABI, CONTRACT_ADDRESS);
    return await web3.eth.getChainId() === 11155111; // Sepolia chain ID
}

// Update Hero Stats
function updateStats(anomalyCount = 0, blockchainCount = 0) {
    document.getElementById('totalAnomalies').textContent = anomalyCount;
    document.getElementById('totalLogs').textContent = blockchainCount;
}

// Anomaly Log Functionsasync function fetchAnomalies() {
async function fetchAnomalies() {
    const loading = document.getElementById('anomalyLoading');
    const list = document.getElementById('anomalyList');
    const error = document.getElementById('anomalyError');
    const noAnomalies = document.getElementById('noAnomalies');
    const errorMsg = document.getElementById('anomalyErrorMsg');

    // Ensure initial state
    loading.classList.remove('d-none');
    list.innerHTML = '';
    error.classList.add('d-none');
    noAnomalies.classList.add('d-none');

    try {
        const response = await fetch(`${RPI_BASE_URL}/api/rpi/history`, {
            headers: {
                'ngrok-skip-browser-warning': 'true'
            }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            throw new Error(`Invalid response (HTML detected): ${text.substring(0, 100)}...`);
        }
        
        const data = await response.json();
        console.log('Anomaly data received:', data); // Debug raw data
        if (!data.success) throw new Error(data.error || 'Unknown error');

        const anomalyCount = data.history.length;
        updateStats(anomalyCount);  // Update hero stat
        console.log('Anomaly count:', anomalyCount); // Debug count

        if (anomalyCount === 0) {
            console.log('No anomalies to display');
            noAnomalies.classList.remove('d-none');
            return;
        }

        let validEntries = 0;
        data.history.forEach((item, index) => {
            // Validate required fields
            if (!item.timestamp || !item.score || !item.status || (!item.frame_path && !item.filename)) {
                console.warn(`Skipping invalid anomaly entry at index ${index}:`, item);
                return;
            }

            const imageName = item.filename || (item.frame_path ? item.frame_path.split('/').pop() : '');
            const imageUrl = imageName ? `${RPI_BASE_URL}/api/rpi/image/${encodeURIComponent(imageName)}` : 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';

            const col = document.createElement('div');
            col.className = 'col-md-6 col-lg-4';
            col.innerHTML = `
                <div class="card anomaly-card h-100">
                    <img src="${imageUrl}" 
                         class="card-img-top" alt="Anomaly Frame" loading="lazy" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg=='">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">${item.timestamp || 'N/A'}</h6>
                        <p class="card-text">
                            <strong>Score:</strong> <span class="badge ${item.status === 'ANOMALY' ? 'bg-danger' : 'bg-success'}">${item.score || 'N/A'}</span><br>
                            <strong>Status:</strong> ${item.status || 'N/A'}<br>
                            <strong>Hash:</strong> ${item.frame_hash ? item.frame_hash.substring(0, 16) : 'N/A'}...
                        </p>
                    </div>
                </div>
            `;
            list.appendChild(col);
            validEntries++;
        });

        console.log(`Rendered ${validEntries} valid anomaly entries`);

        if (validEntries === 0 && anomalyCount > 0) {
            noAnomalies.classList.remove('d-none');
            console.warn('No valid entries rendered despite non-zero anomaly count');
        }
    } catch (err) {
        errorMsg.textContent = `Failed to fetch anomalies: ${err.message}`;
        error.classList.remove('d-none');
        console.error('Fetch error:', err);
    } finally {
        loading.classList.add('d-none');
    }
}

// ---- LIVE FEED ----
function setupLiveFeed() {
    const video = document.getElementById('liveVideo');
    const loading = document.getElementById('videoLoading');
    const error   = document.getElementById('videoError');
    const errorMsg= document.getElementById('videoErrorMsg');

    if (video.src) return;

    // THE ONLY LINE THAT MATTERS
    const streamUrl = `${RPI_BASE_URL}/video_feed?ngrok-skip-browser-warning=1&t=${Date.now()}`;

    video.src = streamUrl;
    console.log('MJPEG →', streamUrl);

    video.onload = () => {
        loading.classList.add('d-none');
        video.style.display = 'block';
        error.classList.add('d-none');
    };
    video.onerror = (e) => {
        loading.classList.add('d-none');
        errorMsg.textContent = `MJPEG failed – open DevTools → Network → check response`;
        error.classList.remove('d-none');
        console.error('MJPEG error', e);
    };
}

// Blockchain Logs Functions
async function fetchBlockchainLogs() {
    const loading = document.getElementById('blockchainLoading');
    const list = document.getElementById('blockchainList');
    const countEl = document.getElementById('blockchainCount');
    const error = document.getElementById('blockchainError');
    const noLogs = document.getElementById('noBlockchainLogs');
    const errorMsg = document.getElementById('blockchainErrorMsg');

    loading.classList.remove('d-none');
    list.innerHTML = '';
    error.classList.add('d-none');
    noLogs.classList.add('d-none');

    try {
        const count = await contract.methods.getAnomalyCount().call();
        const blockchainCount = Number(count);
        countEl.innerHTML = `<h5 class="text-success"><i class="fas fa-lock me-2"></i>Total Logs: ${blockchainCount}</h5>`;
        updateStats(document.getElementById('totalAnomalies').textContent, blockchainCount);  // Update hero stat

        if (blockchainCount === 0) {
            noLogs.classList.remove('d-none');
            return;
        }

        let logs = [];
        for (let i = 0; i < blockchainCount; i++) {
            const anomaly = await contract.methods.getAnomaly(i).call();
            logs.push({
                index: i,
                folder: anomaly[0], // Timestamp
                frameIdx: anomaly[1].toString(), // Hash as string
                error: anomaly[2] // Score
            });
        }

        // Sort by index (newest first)
        logs.reverse();

        const table = `
            <table class="table table-striped">
                <thead><tr><th>Index</th><th>Timestamp (Folder)</th><th>Frame Hash (Idx)</th><th>Error Score</th></tr></thead>
                <tbody>
                    ${logs.map(log => `
                        <tr>
                            <td>${log.index}</td>
                            <td>${log.folder}</td>
                            <td>${log.frameIdx.substring(0, 16)}...</td>
                            <td><span class="badge bg-warning">${log.error}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        list.innerHTML = table;
    } catch (err) {
        errorMsg.textContent = `Blockchain query failed: ${err.message}`;
        error.classList.remove('d-none');
    } finally {
        loading.classList.add('d-none');
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', async () => {
    await initWeb3();
    fetchAnomalies();  // Initial load for stats
    fetchBlockchainLogs();  // Initial load for stats

    // Tab button clicks: Open modals
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const modalElement = document.getElementById(targetId);
            const modal = new bootstrap.Modal(modalElement);
            modal.show();

            // Load content on modal open
            modalElement.addEventListener('shown.bs.modal', () => {
                if (targetId === 'live-feed-modal') {
                    setupLiveFeed();  // Setup video only when opened
                } else if (targetId === 'anomaly-log-modal') {
                    fetchAnomalyInModal();  // Fetch fresh data
                } else if (targetId === 'blockchain-logs-modal') {
                    fetchBlockchainInModal();  // Fetch fresh data
                }
            }, { once: true });  // Run once per open
        });
    });

    // Manual refreshes (inside modals)
    document.getElementById('refreshAnomaly').addEventListener('click', () => {
        fetchAnomalyInModal();  // Wrapper to update stats too
    });
    document.getElementById('refreshBlockchain').addEventListener('click', () => {
        fetchBlockchainInModal();  // Wrapper to update stats too
    });

    // Auto-poll (runs in background, updates stats)
    setInterval(() => {
        fetchAnomalies();  // Just for stats—no modal needed
        fetchBlockchainLogs();  // Just for stats—no modal needed
    }, POLL_INTERVAL);
});

// Wrapper functions for modal refreshes (update stats)
async function fetchAnomalyInModal() {
    await fetchAnomalies();
}

async function fetchBlockchainInModal() {
    await fetchBlockchainLogs();
}
