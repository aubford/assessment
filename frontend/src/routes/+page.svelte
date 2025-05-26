<script lang="ts">
  import { onMount } from "svelte"
  import { Button } from "$lib/components/ui/button"
  import { flyAndScale } from "$lib/utils"

  const apiUrl = "http://localhost:8000"

  interface Order {
    id: number
    burgers: number
    fries: number
    drinks: number
    timestamp: string
  }

  interface OrderData {
    orders: Order[]
  }

  let message = ""
  let orders: Order[] = []
  let isLoading = false
  let errorMessage = ""

  // Computed totals from orders
  $: totals = orders.reduce(
    (acc, order) => ({
      burgers: acc.burgers + order.burgers,
      fries: acc.fries + order.fries,
      drinks: acc.drinks + order.drinks,
    }),
    { burgers: 0, fries: 0, drinks: 0 }
  )

  async function fetchOrders() {
    try {
      const response = await fetch(`${apiUrl}/orders`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data: OrderData = await response.json()
      orders = data.orders
    } catch (error) {
      console.error("Error fetching orders:", error)
      errorMessage = "Failed to fetch orders. Please try again."
    }
  }

  async function submitOrder() {
    if (!message.trim()) return

    isLoading = true
    errorMessage = ""
    try {
      const response = await fetch(`${apiUrl}/process_order`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: message.trim() }),
      })

      if (response.ok) {
        message = ""
        await fetchOrders()
      } else {
        const errorData = await response
          .json()
          .catch(() => ({ detail: "Unknown error occurred" }))
        errorMessage = errorData.detail || "Error processing order"
      }
    } catch (error) {
      console.error("Error submitting order:", error)
      errorMessage = "Failed to submit order. Please check your connection and try again."
    } finally {
      isLoading = false
    }
  }

  function formatOrderItems(order: Order): string {
    const items = []
    if (order.burgers > 0) {
      items.push(`${order.burgers} Burger${order.burgers > 1 ? "s" : ""}`)
    }
    if (order.fries > 0) {
      items.push(`${order.fries} Fr${order.fries > 1 ? "ies" : "y"}`)
    }
    if (order.drinks > 0) {
      items.push(`${order.drinks} Drink${order.drinks > 1 ? "s" : ""}`)
    }
    return items.join(", ")
  }

  function formatTimestamp(timestamp: string): string {
    return new Date(timestamp).toLocaleString()
  }

  function clearError() {
    errorMessage = ""
  }

  onMount(() => {
    fetchOrders()
  })
</script>

<!-- Error Message -->
{#if errorMessage}
  <div
    class="bg-red-50 border border-red-200 rounded-lg p-4 flex justify-between items-center"
  >
    <div class="flex items-center">
      <span class="text-red-600 mr-3">⚠️</span>
      <span class="text-red-800">{errorMessage}</span>
    </div>
    <Button
      variant="ghost"
      size="sm"
      on:click={clearError}
      class="text-red-600 hover:text-red-800 h-6 w-6 p-0">×</Button
    >
  </div>
{/if}

<!-- Totals Section -->
<div class="grid grid-cols-3 gap-6">
  {#each [{ item: "burgers", value: totals.burgers }, { item: "fries", value: totals.fries }, { item: "drinks", value: totals.drinks }] as total}
    <div class="bg-white rounded-lg border-2 border-gray-300 p-6 text-center">
      <h2 class="text-lg font-medium text-gray-700 mb-2">Total # of {total.item}s</h2>
      <div class="text-3xl font-bold text-gray-900">{total.value}</div>
    </div>
  {/each}
</div>

<!-- Order Input Section -->
<div class="bg-white rounded-lg border-2 border-gray-300 p-6">
  <div class="flex gap-4">
    <div class="flex-1">
      <label for="message" class="block text-sm font-medium text-gray-700 mb-2">
        Drive thru message:
      </label>
      <div class="text-sm text-gray-500 mb-3">
        Ex: "I would like one burger and an order of fries", "Cancel order #2"
      </div>
      <input
        id="message"
        type="text"
        bind:value={message}
        on:keydown={e => e.key === "Enter" && submitOrder()}
        placeholder="Enter your order..."
        class="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        disabled={isLoading}
      />
    </div>
    <div class="flex items-end">
      <Button
        on:click={submitOrder}
        disabled={isLoading || !message.trim()}
        variant="secondary"
        class="rounded-full border-2 border-gray-300 px-6 h-12"
      >
        {isLoading ? "Processing..." : "run"}
      </Button>
    </div>
  </div>
</div>

<!-- Order History Section -->
<div class="space-y-4">
  <h2 class="text-xl font-medium text-gray-900">Order History</h2>
  <div class="space-y-3">
    {#each orders as order (order.id)}
      <div
        class="bg-white rounded-lg border-2 border-gray-300 p-4 flex justify-between items-start"
        in:flyAndScale
      >
        <div>
          <span class="font-medium text-gray-900">Order #{order.id}</span>
          <div class="text-sm text-gray-500 mt-1">{formatTimestamp(order.timestamp)}</div>
        </div>
        <span class="text-gray-600">{formatOrderItems(order)}</span>
      </div>
    {:else}
      <div
        class="bg-white rounded-lg border-2 border-gray-300 p-4 text-center text-gray-500"
      >
        No orders yet
      </div>
    {/each}
  </div>
</div>
