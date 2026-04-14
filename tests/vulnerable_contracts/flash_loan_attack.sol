// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ──────────────────────────────────────────────────────────────────────────────
// SENTINEL TEST SUITE — POSITIVE TEST #3: Flash Loan Price Oracle Manipulation
// Classification: DeFi Economic Attack / Oracle Manipulation
// Real-world analogues: Harvest Finance ($34M, 2020), Cheese Bank ($3.3M, 2020)
// ──────────────────────────────────────────────────────────────────────────────
// This AMM-based lending protocol calculates the USD price of collateral using
// the SPOT PRICE from its own internal AMM reserve ratio (x*y=k).
// An attacker can:
//   1. Flash-borrow a massive amount of tokenA.
//   2. Dump tokenA into the AMM pool → price of tokenA plummets.
//   3. Borrow against the now-inflated tokenB collateral at manipulated price.
//   4. Repay the flash loan, keeping the profit.
// ──────────────────────────────────────────────────────────────────────────────

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IFlashLoanProvider {
    function flashLoan(address token, uint256 amount, bytes calldata data) external;
}

/// @title VulnerableAMM — a simple constant-product AMM pool.
contract VulnerableAMM {
    IERC20 public tokenA;
    IERC20 public tokenB;

    uint256 public reserveA;
    uint256 public reserveB;

    constructor(address _tokenA, address _tokenB) {
        tokenA = IERC20(_tokenA);
        tokenB = IERC20(_tokenB);
    }

    /// @notice Seed initial liquidity.
    function addLiquidity(uint256 amountA, uint256 amountB) external {
        tokenA.transferFrom(msg.sender, address(this), amountA);
        tokenB.transferFrom(msg.sender, address(this), amountB);
        reserveA += amountA;
        reserveB += amountB;
    }

    /// @notice Swap tokenA for tokenB using constant-product formula.
    function swapAforB(uint256 amountAIn) external returns (uint256 amountBOut) {
        tokenA.transferFrom(msg.sender, address(this), amountAIn);
        // k = reserveA * reserveB must remain constant
        uint256 newReserveA = reserveA + amountAIn;
        uint256 newReserveB = (reserveA * reserveB) / newReserveA;
        amountBOut = reserveB - newReserveB;
        reserveA = newReserveA;
        reserveB = newReserveB;
        tokenB.transfer(msg.sender, amountBOut);
    }

    // VULNERABILITY: Spot price derived directly from live reserve ratio.
    // This function is the oracle consumed by the lending protocol below.
    // It reflects the instantaneous reserve state — manipulable in the same tx.
    function getSpotPriceAinB() external view returns (uint256) {
        require(reserveA > 0, "No liquidity");
        // Price of 1 tokenA expressed in tokenB units (scaled by 1e18)
        return (reserveB * 1e18) / reserveA;
    }
}


/// @title VulnerableLendingProtocol — borrows against collateral priced via AMM spot.
contract VulnerableLendingProtocol {
    VulnerableAMM public amm;
    IERC20 public collateralToken; // tokenB
    IERC20 public borrowToken;    // tokenA

    mapping(address => uint256) public collateralDeposited;
    mapping(address => uint256) public debtOutstanding;

    uint256 public constant COLLATERAL_FACTOR = 75; // 75% LTV

    constructor(address _amm, address _collateral, address _borrow) {
        amm = VulnerableAMM(_amm);
        collateralToken = IERC20(_collateral);
        borrowToken = IERC20(_borrow);
    }

    function depositCollateral(uint256 amount) external {
        collateralToken.transferFrom(msg.sender, address(this), amount);
        collateralDeposited[msg.sender] += amount;
    }

    // VULNERABILITY: maxBorrow computed using the live AMM spot price.
    // Flash loan manipulation of reserves directly inflates this value.
    function borrow(uint256 borrowAmount) external {
        uint256 collateral = collateralDeposited[msg.sender];
        require(collateral > 0, "No collateral");

        // Spot price: how many tokenA is 1 tokenB worth right now?
        uint256 spotPrice = amm.getSpotPriceAinB();   // <-- MANIPULATED ORACLE

        // Max borrowable tokenA given posted tokenB collateral
        uint256 collateralValueInA = (collateral * spotPrice) / 1e18;
        uint256 maxBorrow = (collateralValueInA * COLLATERAL_FACTOR) / 100;

        require(
            debtOutstanding[msg.sender] + borrowAmount <= maxBorrow,
            "Borrow exceeds collateral"
        );

        debtOutstanding[msg.sender] += borrowAmount;
        borrowToken.transfer(msg.sender, borrowAmount);
    }
}
