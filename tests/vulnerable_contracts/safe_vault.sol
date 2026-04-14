// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ──────────────────────────────────────────────────────────────────────────────
// SENTINEL TEST SUITE — NEGATIVE TEST #1: Secure ETH Vault (No Known Bugs)
// ──────────────────────────────────────────────────────────────────────────────
// This contract is intentionally SECURE and represents production-quality code.
// It is used to test the model's False Positive rate.
// Key security properties implemented:
//   ✅ Checks-Effects-Interactions (CEI) pattern throughout
//   ✅ ReentrancyGuard on all state-mutating external functions
//   ✅ msg.sender-based auth (no tx.origin anywhere)
//   ✅ Two-step ownership transfer (no abrupt privilege handoff)
//   ✅ Solidity 0.8.x (automatic overflow/underflow revert)
//   ✅ No unprotected initialisation functions
//   ✅ Emergency pause via circuit breaker
//   ✅ No delegatecall, no assembly, no selfdestruct
// ──────────────────────────────────────────────────────────────────────────────

/// @notice Minimal reentrancy guard — CEI + mutex.
abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status = _NOT_ENTERED;

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

/// @notice Circuit breaker for emergency pause.
abstract contract Pausable {
    bool private _paused;
    address internal _admin;

    event Paused(address account);
    event Unpaused(address account);

    modifier whenNotPaused() {
        require(!_paused, "Pausable: contract is paused");
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == _admin, "Pausable: caller is not admin");
        _;
    }

    function paused() public view returns (bool) {
        return _paused;
    }

    function pause() external onlyAdmin {
        _paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyAdmin {
        _paused = false;
        emit Unpaused(msg.sender);
    }
}


/// @title SecureVault — a production-quality ETH custody contract.
/// @notice Demonstrates CEI, ReentrancyGuard, two-step ownership, and pause.
contract SecureVault is ReentrancyGuard, Pausable {

    // ── State ─────────────────────────────────────────────────────────────────
    mapping(address => uint256) private _balances;
    address public pendingAdmin;

    uint256 public totalDeposits;
    uint256 public constant MAX_DEPOSIT = 100 ether;

    // ── Events ────────────────────────────────────────────────────────────────
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event AdminTransferProposed(address indexed current, address indexed candidate);
    event AdminTransferAccepted(address indexed newAdmin);

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor() {
        _admin = msg.sender;
    }

    // ── Core Vault Logic ──────────────────────────────────────────────────────

    /// @notice Deposit ETH into the vault.
    /// @dev Enforces a per-user max deposit to limit protocol exposure.
    function deposit() external payable whenNotPaused {
        require(msg.value > 0, "Vault: zero deposit");
        require(
            _balances[msg.sender] + msg.value <= MAX_DEPOSIT,
            "Vault: deposit exceeds maximum"
        );

        // EFFECT before any interaction (CEI)
        _balances[msg.sender] += msg.value;
        totalDeposits += msg.value;

        emit Deposited(msg.sender, msg.value);
        // No external call after state change — pure CEI.
    }

    /// @notice Withdraw ETH from the vault.
    /// @dev Full CEI: state zeroed BEFORE the external call; mutex guards re-entry.
    function withdraw(uint256 amount) external nonReentrant whenNotPaused {
        require(amount > 0, "Vault: zero withdrawal");
        require(_balances[msg.sender] >= amount, "Vault: insufficient balance");

        // CHECK  → already done above
        // EFFECT → state update first, before any external call
        _balances[msg.sender] -= amount;
        totalDeposits -= amount;

        // INTERACTION → called only after full state update
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Vault: ETH transfer failed");

        emit Withdrawn(msg.sender, amount);
    }

    // ── View Functions ────────────────────────────────────────────────────────

    /// @notice Returns the caller's vault balance.
    function balanceOf() external view returns (uint256) {
        return _balances[msg.sender];
    }

    /// @notice Returns the vault's total ETH balance.
    function vaultBalance() external view returns (uint256) {
        return address(this).balance;
    }

    // ── Two-Step Ownership Transfer ───────────────────────────────────────────

    /// @notice Propose a new admin. Does NOT transfer power immediately.
    /// @dev Two-step pattern prevents accidental loss of admin access.
    function proposeAdmin(address candidate) external onlyAdmin {
        require(candidate != address(0), "Vault: zero address");
        pendingAdmin = candidate;
        emit AdminTransferProposed(_admin, candidate);
    }

    /// @notice New admin accepts their role. Old admin loses access atomically.
    function acceptAdmin() external {
        require(msg.sender == pendingAdmin, "Vault: not pending admin");
        _admin = pendingAdmin;
        pendingAdmin = address(0);
        emit AdminTransferAccepted(_admin);
    }

    // ── Fallback Guard ────────────────────────────────────────────────────────

    /// @dev Reject plain ETH sends — deposits must go through `deposit()`.
    receive() external payable {
        revert("Vault: use deposit()");
    }
}
