<script lang="ts">
  import "../app.css"
  let { children } = $props()
</script>

<div class="min-h-screen bg-gray-50 p-8">
  <div class="max-w-4xl mx-auto space-y-8">
    {@render children()}
  </div>
</div>
