import os
from web3 import Web3

# 1. Setup Connection
INFURA_URL = "https://sepolia.infura.io/v3/8742554fd5c94c549cb8b4117b076e7a"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# 2. Contract Details
CONTRACT_ADDRESS = "0x279FcACc1eB244BBD7Be138D34F3f562Da179dd5"
# Paste the ABI you shared earlier here
CONTRACT_ABI = [
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "_folder",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "_frameIdx",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "_error",
				"type": "string"
			}
		],
		"name": "logAnomaly",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"stateMutability": "nonpayable",
		"type": "constructor"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "anomalies",
		"outputs": [
			{
				"internalType": "string",
				"name": "folder",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "frameIdx",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "error",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "index",
				"type": "uint256"
			}
		],
		"name": "getAnomaly",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "getAnomalyCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "owner",
		"outputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
] 

# 3. Security (Load from Environment)
PRIVATE_KEY = os.getenv("METAMASK_PRIVATE_KEY") 
WALLET_ADDRESS = "0xa8824b2E3b176bBc530a6a6B54f08beb0447C21e"

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

def log_bandit_attack_to_blockchain(folder_name, frame_idx, error_msg):
    """Triggers the Smart Contract logAnomaly function"""
    
    # Check connection
    if not w3.is_connected():
        print("Error: Not connected to Sepolia")
        return

    # Build transaction
    nonce = w3.eth.get_transaction_count(WALLET_ADDRESS)
    
    txn = contract.functions.logAnomaly(
        folder_name, 
        frame_idx, 
        error_msg
    ).build_transaction({
        'chainId': 11155111, # Sepolia Chain ID
        'gas': 200000,
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })

    # Sign and Send
    signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
    txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    
    print(f"Attack logged to Blockchain! Hash: {w3.to_hex(txn_hash)}")
    return w3.to_hex(txn_hash)
