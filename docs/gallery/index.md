# Plot Gallery

OpenRetailScience provides a comprehensive set of plotting functions designed specifically for retail analytics. All plots
use a consistent API and come pre-styled with retail-friendly color schemes and professional styling.

## Plot Types

<!-- markdownlint-disable MD033 -->
<style>
/* Matplotlib-style gallery container */
.glr-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, 200px);
  gap: 15px;
  margin: 20px 0;
  justify-content: start;
}

/* Individual thumbnail container - consistent card style */
.glr-thumbcontainer {
  text-decoration: none;
  color: inherit;
  display: block;
  text-align: center;
  transition: transform 0.2s ease;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 12px;
  background: var(--md-default-bg-color, #fff);
  width: 200px;
}

.glr-thumbcontainer:hover {
  transform: scale(1.02);
  text-decoration: none;
  border-color: #007acc;
  box-shadow: 0 2px 8px rgba(0, 122, 204, 0.15);
}

/* Card header - plot type name */
.glr-thumb-title {
  font-size: 14px;
  margin-bottom: 12px;
  font-weight: 600;
  color: var(--md-default-fg-color, #333);
  line-height: 1.2;
  border-bottom: 1px solid var(--md-default-fg-color--lightest, #eee);
  padding-bottom: 8px;
}

/* Thumbnail image */
.glr-thumb img {
  width: 160px;
  height: 112px;
  object-fit: contain;
  border: 1px solid #ddd;
  border-radius: 4px;
  display: block;
  margin: 0 auto;
  background: #fff;
  padding: 4px;
}

/* Remove image-specific hover effects - only hover on outer container */

/* Coming soon placeholder */
.coming-soon-thumb {
  width: 160px;
  height: 112px;
  background: #f0f0f0;
  border: 1px dashed #ccc;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: #999;
  margin: 0 auto;
}

.coming-soon-container {
  color: #666;
  cursor: default;
  border: 1px solid #ddd;
  background: var(--md-default-bg-color, #f9f9f9);
}

.coming-soon-container:hover {
  transform: none;
  border-color: #ddd;
  box-shadow: none;
}

.coming-soon-title {
  font-size: 14px;
  margin-bottom: 12px;
  font-weight: 600;
  color: var(--md-default-fg-color--light, #666);
  line-height: 1.2;
  border-bottom: 1px solid var(--md-default-fg-color--lightest, #ddd);
  padding-bottom: 8px;
}
</style>

<div class="glr-gallery">
  <a href="plots/area/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Area Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/area_thumbnail.png" alt="Area Plot">
    </div>
  </a>

  <a href="plots/bar/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Bar Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/bar_thumbnail.png" alt="Bar Plot">
    </div>
  </a>
  <a href="plots/broken_timeline/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Broken Timeline Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/broken_timeline_thumbnail.png" alt="Broken Timeline Plot">
    </div>
  </a>
  <a href="plots/cohort/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Cohort Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/cohort_thumbnail.png" alt="Cohort Plot">
    </div>
  </a>
  <a href="plots/customer_decision_hierarchy/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Customer Decision Hierarchy</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/customer_decision_hierarchy_thumbnail.png" alt="Customer Decision Hierarchy">
    </div>
  </a>
  <a href="plots/heatmap/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Heatmap Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/heatmap_thumbnail.png" alt="Heatmap Plot">
    </div>
  </a>
  <a href="plots/histogram/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Histogram Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/histogram_thumbnail.png" alt="Histogram Plot">
    </div>
  </a>
  <a href="plots/line/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Line Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/line_thumbnail.png" alt="Line Plot">
    </div>
  </a>

  <a href="plots/period_on_period/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Period on Period Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/period_on_period_thumbnail.png" alt="Period on Period Plot">
    </div>
    </a>

  <a href="plots/revenue_tree/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Revenue Tree</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/revenue_tree_thumbnail.png" alt="Revenue Tree">
    </div>
  </a>

  <a href="plots/time/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Time Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/time_thumbnail.png" alt="Time Plot">
    </div>
  </a>

  <a href="plots/venn/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Venn Diagram</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/venn_thumbnail.png" alt="Venn Diagram">
    </div>
  </a>

  <a href="plots/waterfall/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Waterfall Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/waterfall_thumbnail.png" alt="Waterfall Plot">
    </div>
  </a>

  <a href="plots/index_plot/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Index Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/index_thumbnail.png" alt="Index Plot">
    </div>
  </a>

  <a href="plots/scatter/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Scatter Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/scatter_thumbnail.png" alt="Scatter Plot">
    </div>
  </a>

  <a href="plots/price/" class="glr-thumbcontainer">
    <div class="glr-thumb-title">Price Plot</div>
    <div class="glr-thumb">
      <img src="../assets/images/gallery/price_thumbnail.png" alt="Price Plot">
    </div>
  </a>

</div>
