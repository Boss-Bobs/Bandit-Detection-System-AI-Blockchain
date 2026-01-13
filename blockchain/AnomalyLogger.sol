// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AnomalyLogger {
    struct Anomaly {
        uint256 frameIndex;
        string anomalyType;
        uint256 timestamp;
    }

    Anomaly[] public anomalies;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function logAnomaly(uint256 _frameIndex, string memory _anomalyType) public {
        anomalies.push(Anomaly(_frameIndex, _anomalyType, block.timestamp));
    }

    function batchLogAnomalies(uint256[] memory _frameIndices, string[] memory _anomalyTypes) public {
        require(_frameIndices.length == _anomalyTypes.length, "Arrays must have same length");
        for (uint256 i = 0; i < _frameIndices.length; i++) {
            anomalies.push(Anomaly(_frameIndices[i], _anomalyTypes[i], block.timestamp));
        }
    }

    function getAnomalyCount() public view returns (uint256) {
        return anomalies.length;
    }

    function getAnomaly(uint256 index) public view returns (uint256, string memory, uint256) {
        require(index < anomalies.length, "Index out of bounds");
        Anomaly memory anomaly = anomalies[index];
        return (anomaly.frameIndex, anomaly.anomalyType, anomaly.timestamp);
    }
}
