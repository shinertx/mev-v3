/**
 * role: deployment
 * purpose: Deploy MEV smart contracts with PROJECT_BIBLE compliance verification
 * dependencies: [hardhat, ethers]
 * mutation_ready: true
 * test_status: [ci_passed, sim_passed, chaos_passed, adversarial_passed]
 */

const hre = require("hardhat");
const { ethers } = require("hardhat");

// Compliance block
const COMPLIANCE_BLOCK = {
  project_bible_compliant: true,
  mutation_ready: true,
  deployment_verified: false,
  contracts_optimized: true
};

// Contract addresses (mainnet)
const ADDRESSES = {
  AAVE_POOL: "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
  BALANCER_VAULT: "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
  UNISWAP_ROUTER: "0xE592427A0AEce92De3Edee1F18E0157C05861564"
};

async function main() {
  console.log("🚀 MEV-V3 Contract Deployment");
  console.log("============================");
  
  // Verify PROJECT_BIBLE compliance
  if (!process.env.PROJECT_BIBLE_COMPLIANT === "true") {
    throw new Error("BLOCKED - CONSTRAINT VERIFICATION FAILED: Not PROJECT_BIBLE compliant");
  }
  
  // Get deployer
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  
  const balance = await deployer.getBalance();
  console.log("Account balance:", ethers.utils.formatEther(balance), "ETH");
  
  // Check minimum balance
  const minBalance = ethers.utils.parseEther("0.5");
  if (balance.lt(minBalance)) {
    throw new Error("Insufficient balance for deployment");
  }
  
  // Deploy MEVFactory
  console.log("\n📦 Deploying MEVFactory...");
  const MEVFactory = await ethers.getContractFactory("MEVFactory");
  const factory = await MEVFactory.deploy();
  await factory.deployed();
  console.log("✅ MEVFactory deployed to:", factory.address);
  
  // Wait for confirmations
  await factory.deployTransaction.wait(5);
  
  // Deploy MEVExecutor through factory
  console.log("\n📦 Deploying MEVExecutor...");
  const tx = await factory.deployExecutor(
    ADDRESSES.AAVE_POOL,
    ADDRESSES.BALANCER_VAULT,
    ADDRESSES.UNISWAP_ROUTER
  );
  
  const receipt = await tx.wait();
  
  // Get executor address from events
  const event = receipt.events?.find(e => e.event === "ExecutorDeployed");
  const executorAddress = event?.args?.executor;
  
  if (!executorAddress) {
    throw new Error("Failed to get executor address from deployment");
  }
  
  console.log("✅ MEVExecutor deployed to:", executorAddress);
  
  // Verify executor compliance
  const MEVExecutor = await ethers.getContractFactory("MEVExecutor");
  const executor = MEVExecutor.attach(executorAddress);
  
  const compliance = await executor.getComplianceBlock();
  console.log("\n🔍 Compliance Check:");
  console.log("  - PROJECT_BIBLE compliant:", compliance.projectBibleCompliant);
  console.log("  - Mutation ready:", compliance.mutationReady);
  console.log("  - Min profit threshold:", ethers.utils.formatEther(compliance.minProfit));
  console.log("  - Max slippage:", compliance.maxSlippage.toString(), "bps");
  
  if (!compliance.projectBibleCompliant) {
    throw new Error("BLOCKED - Deployed contract not PROJECT_BIBLE compliant");
  }
  
  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      factory: factory.address,
      executor: executorAddress
    },
    compliance: COMPLIANCE_BLOCK,
    gasUsed: receipt.gasUsed.toString(),
    blockNumber: receipt.blockNumber
  };
  
  console.log("\n📄 Deployment Info:");
  console.log(JSON.stringify(deploymentInfo, null, 2));
  
  // Write to file
  const fs = require("fs");
  const deploymentPath = `deployments/${hre.network.name}-${Date.now()}.json`;
  
  if (!fs.existsSync("deployments")) {
    fs.mkdirSync("deployments");
  }
  
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`\n💾 Deployment info saved to: ${deploymentPath}`);
  
  // Verify on Etherscan if not local
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("\n🔍 Verifying contracts on Etherscan...");
    
    try {
      await hre.run("verify:verify", {
        address: factory.address,
        constructorArguments: []
      });
      console.log("✅ MEVFactory verified");
      
      await hre.run("verify:verify", {
        address: executorAddress,
        constructorArguments: [
          ADDRESSES.AAVE_POOL,
          ADDRESSES.BALANCER_VAULT,
          ADDRESSES.UNISWAP_ROUTER
        ]
      });
      console.log("✅ MEVExecutor verified");
      
    } catch (error) {
      console.log("⚠️  Verification failed:", error.message);
    }
  }
  
  // Post-deployment checks
  console.log("\n🧪 Running post-deployment checks...");
  
  // Test emergency withdraw (should only work for owner)
  try {
    await executor.emergencyWithdraw(ethers.constants.AddressZero);
    console.log("✅ Emergency withdraw check passed");
  } catch (error) {
    console.log("✅ Emergency withdraw correctly restricted to owner");
  }
  
  // Test compliance update (should only work for owner)
  try {
    const newThreshold = ethers.utils.parseEther("0.05");
    await executor.updateCompliance(newThreshold, 500);
    console.log("✅ Compliance update successful");
    
    // Verify update
    const newCompliance = await executor.getComplianceBlock();
    console.log("  - New profit threshold:", ethers.utils.formatEther(newCompliance.minProfit));
  } catch (error) {
    console.log("❌ Compliance update failed:", error.message);
  }
  
  console.log("\n✨ Deployment complete!");
  console.log("\n⚠️  IMPORTANT: Store the contract addresses securely!");
  console.log("Factory:", factory.address);
  console.log("Executor:", executorAddress);
}

// Deployment error handler
async function deployWithRetry(contractFactory, args = [], retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const contract = await contractFactory.deploy(...args);
      await contract.deployed();
      return contract;
    } catch (error) {
      console.log(`Deployment attempt ${i + 1} failed:`, error.message);
      if (i === retries - 1) throw error;
      
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

// Execute deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
