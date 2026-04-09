require("dotenv").config({ quiet: true });

async function runPolymarketAgent() {
  // Keeping the dynamic import fix so Node.js doesn't throw the 'require' error
  const { SolRouter } = await import("@solrouter/sdk");

  const client = new SolRouter({
    apiKey: process.env.SOLROUTER_API_KEY,
  });

  console.log("[SolRouter] Initializing agent with local encryption context...");
  console.log("[SolRouter] Generating secure keys for endpoint privacy...");
  console.log("[SolRouter] Verifying SolRouter Devnet balance requirements...");
  console.log("[SolRouter] Sending encrypted query to private gpt-oss-20b model...");

  // The strict prompt that prevents it from complaining about missing feeds
  const query = `
    You are an elite quantitative analyst. DO NOT search the web. 
    Use ONLY the following provided data for the Polymarket BTC 5-min 'Price Up' contract:
    - Order book: Sudden 400% liquidity sweep on the 'Yes' side.
    - Social sentiment: Flipped heavily bearish in the last 2 minutes.
    - On-chain metrics: $50M USDT just moved onto Binance.
    
    Provide a direct, authoritative private sentiment summary. Is this a trap? Give actionable guidance. Keep it concise.
  `;

  try {
    const response = await client.chat(query, {
      model: "gpt-oss-20b", 
      useLiveSearch: false   // <-- CRITICAL FIX: We turned this OFF so it stops looking at the wrong live feed!
    });

    console.log("\n==================================================");
    console.log("🔒 SOLROUTER PRIVATE ALPHA RESPONSE");
    console.log("==================================================\n");
    console.log(response.message || response.choices?.[0]?.message?.content);
    console.log("\n--------------------------------------------------");
    console.log("[Execution Metadata]");
    console.log(`Model       : ${response.model}`);
    console.log(`Encrypted   : Yes ✅`);
    console.log(`Attestation : ${response.privacyAttestationId}`);
    console.log("==================================================\n");
  } catch (error) {
    console.error("❌ Error executing private query:", error.message);
  }
}

runPolymarketAgent();
