// Interactive Navigation Visualizer - Luxury Edition

// Node positions on the canvas (percentages for responsiveness)
const nodePositions = {
    // Ground Floor (Bottom - ~75-90%)
    'R11': { x: 53, y: 79 },
    'R12': { x: 68, y: 79 },
    'S1': { x: 35, y: 79 },
    'R15': { x: 20, y: 79 },
    'R14': { x: 27, y: 88 },
    'R13': { x: 47, y: 88 },
    
    // First Floor (Middle - ~40-55%)
    'S2': { x: 20, y: 47 },
    'R21': { x: 35, y: 47 },
    'S3': { x: 53, y: 47 },
    'R22': { x: 68, y: 47 },
    'R23': { x: 80, y: 47 },
    'R24': { x: 93, y: 47 },
    
    // Second Floor (Top - ~8-22%)
    'R31': { x: 20, y: 15 },
    'R32': { x: 35, y: 15 },
    'R33': { x: 53, y: 15 },
    'S4': { x: 68, y: 15 },
    'R34': { x: 80, y: 15 },
    'R35': { x: 93, y: 15 }
};

// Stairs nodes
const stairsNodes = ['S1', 'S2', 'S3', 'S4'];

/**
 * Initialize the visual navigator
 */
function initializeVisualNavigator() {
    console.log('%c🏎️ QUANTUM PATHFINDER Initialized', 'color: #D4AF37; font-size: 18px; font-weight: bold;');
    
    // Setup form interaction
    setupFormInteraction();
    
    // Initialize canvas visualization if result exists
    initializeCanvasVisualization();
}

/**
 * Setup form interaction and validation
 */
function setupFormInteraction() {
    const form = document.querySelector('form');
    const compareMode = document.getElementById('compare_mode');
    const algorithmSelect = document.getElementById('algorithm');
    
    // Handle compare mode toggle
    if (compareMode && algorithmSelect) {
        compareMode.addEventListener('change', function() {
            algorithmSelect.disabled = this.checked;
            algorithmSelect.style.opacity = this.checked ? '0.5' : '1';
            
            if (this.checked) {
                console.log('🔬 Comparison mode activated');
            }
        });
    }
    
    // Handle form submission
    if (form) {
        form.addEventListener('submit', function(e) {
            const btn = this.querySelector('.launch-btn');
            if (btn) {
                btn.innerHTML = '<span class="loading-spinner"></span> CALCULATING ROUTE...';
                btn.disabled = true;
            }
        });
    }
}

/**
 * Initialize canvas visualization
 */
function initializeCanvasVisualization() {
    const canvas = document.getElementById('pathCanvas');
    if (!canvas) return;
    
    // Create all node markers on canvas
    createCanvasNodes(canvas);
    
    // Create SVG overlay for path drawing
    createSVGOverlay(canvas);
    
    // Highlight path if exists
    const pathDisplay = document.querySelector('.route-path');
    if (pathDisplay && pathDisplay.textContent) {
        const pathString = pathDisplay.textContent.trim();
        visualizePath(pathString, canvas);
    }
}

/**
 * Create visual node markers on the canvas
 */
function createCanvasNodes(canvas) {
    Object.keys(nodePositions).forEach(nodeId => {
        const node = document.createElement('div');
        node.className = 'canvas-node default';
        node.setAttribute('data-node', nodeId);
        node.textContent = nodeId;
        
        // Add stairs class if applicable
        if (stairsNodes.includes(nodeId)) {
            node.classList.add('stairs');
        }
        
        // Position the node
        const pos = nodePositions[nodeId];
        node.style.left = `${pos.x}%`;
        node.style.top = `${pos.y}%`;
        
        // Add tooltip
        node.title = `Node: ${nodeId}`;
        
        // Add click handler
        node.addEventListener('click', function() {
            console.log(`📍 Node clicked: ${nodeId}`);
        });
        
        canvas.appendChild(node);
    });
    
    console.log('✅ Canvas nodes created');
}

/**
 * Create SVG overlay for drawing paths
 */
function createSVGOverlay(canvas) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('canvas-svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    canvas.appendChild(svg);
}

/**
 * Visualize the path on the canvas
 */
function visualizePath(pathString, canvas) {
    if (!pathString || pathString === 'No path found') {
        console.log('❌ No valid path to visualize');
        return;
    }
    
    // Parse the path
    const nodes = pathString.split(' → ').map(n => n.trim()).filter(n => n);
    
    if (nodes.length === 0) {
        console.log('❌ Empty path');
        return;
    }
    
    console.log(`🗺️ Visualizing path: ${nodes.join(' → ')}`);
    
    // Highlight nodes
    highlightPathNodes(nodes);
    
    // Draw path line
    drawPathLine(nodes, canvas);
    
    // Scroll to visual navigator
    setTimeout(() => {
        const visualNav = document.querySelector('.visual-navigator');
        if (visualNav) {
            visualNav.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }, 800);
}

/**
 * Highlight nodes along the path
 */
function highlightPathNodes(nodes) {
    // Reset all nodes
    document.querySelectorAll('.canvas-node').forEach(node => {
        node.classList.remove('start', 'end', 'path');
        if (stairsNodes.includes(node.getAttribute('data-node'))) {
            node.classList.add('stairs');
        } else {
            node.className = 'canvas-node default';
        }
    });
    
    nodes.forEach((nodeId, index) => {
        const node = document.querySelector(`.canvas-node[data-node="${nodeId}"]`);
        if (node) {
            // Remove default class
            node.classList.remove('default');
            
            if (index === 0) {
                // Start node
                node.classList.add('start');
                console.log(`🔵 Start: ${nodeId}`);
            } else if (index === nodes.length - 1) {
                // End node
                node.classList.add('end');
                console.log(`🔴 End: ${nodeId}`);
            } else {
                // Path node
                node.classList.add('path');
                console.log(`🟢 Path: ${nodeId}`);
            }
        }
    });
}

/**
 * Draw animated line connecting path nodes
 */
function drawPathLine(nodes, canvas) {
    const svg = canvas.querySelector('.canvas-svg');
    if (!svg) {
        console.log('⚠️ SVG overlay not found');
        return;
    }
    
    // Clear existing paths
    svg.innerHTML = '';
    
    // Get canvas dimensions
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    // Create path points
    const points = nodes.map(nodeId => {
        const pos = nodePositions[nodeId];
        if (!pos) return null;
        return {
            x: (pos.x / 100) * width,
            y: (pos.y / 100) * height,
            id: nodeId
        };
    }).filter(p => p !== null);
    
    if (points.length < 2) {
        console.log('⚠️ Not enough points to draw path');
        return;
    }
    
    // Create smooth path
    let pathData = `M ${points[0].x} ${points[0].y}`;
    
    for (let i = 1; i < points.length; i++) {
        pathData += ` L ${points[i].x} ${points[i].y}`;
    }
    
    // Create glow effect (background)
    const glowPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    glowPath.setAttribute('d', pathData);
    glowPath.classList.add('path-glow');
    svg.appendChild(glowPath);
    
    // Create main path
    const pathElement = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathElement.setAttribute('d', pathData);
    pathElement.classList.add('path-line');
    
    // Calculate path length for animation
    const pathLength = 1000;
    pathElement.setAttribute('stroke-dasharray', pathLength);
    pathElement.setAttribute('stroke-dashoffset', pathLength);
    
    svg.appendChild(pathElement);
    
    // Add circles at connection points
    points.forEach((point, index) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', point.x);
        circle.setAttribute('cy', point.y);
        circle.setAttribute('r', '6');
        
        // Color based on position
        let fill = '#00D9FF'; // Default cyan
        if (index === 0) fill = '#00D9FF'; // Start - blue
        else if (index === points.length - 1) fill = '#E31837'; // End - red
        
        circle.setAttribute('fill', fill);
        circle.setAttribute('opacity', '0.8');
        circle.style.filter = 'drop-shadow(0 0 8px ' + fill + ')';
        
        svg.appendChild(circle);
    });
    
    console.log('✨ Path visualization complete!');
}

/**
 * Add interactive effects to comparison table
 */
function enhanceComparisonTable() {
    const rows = document.querySelectorAll('.leaderboard-table tbody tr');
    rows.forEach((row, index) => {
        row.style.animationDelay = `${index * 0.05}s`;
    });
}

/**
 * Initialize all features when DOM is ready
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeVisualNavigator();
    enhanceComparisonTable();
    
    console.log('%c⚡ All systems operational', 'color: #00D9FF; font-size: 14px;');
    console.log('%c🏁 Ready to navigate!', 'color: #D4AF37; font-size: 14px;');
});

/**
 * Handle window resize to redraw path
 */
let resizeTimeout;
window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
        const pathDisplay = document.querySelector('.route-path');
        const canvas = document.getElementById('pathCanvas');
        
        if (pathDisplay && canvas && pathDisplay.textContent) {
            const pathString = pathDisplay.textContent.trim();
            const nodes = pathString.split(' → ').map(n => n.trim()).filter(n => n);
            
            if (nodes.length > 0) {
                // Clear old SVG
                const svg = canvas.querySelector('.canvas-svg');
                if (svg) svg.innerHTML = '';
                
                // Redraw
                drawPathLine(nodes, canvas);
            }
        }
    }, 250);
});
