document.addEventListener("DOMContentLoaded", () => {
    // Select <figure> elements in <body>, excluding those inside <d-title>
    const figures = document.querySelectorAll("body figure:not(d-title figure)");
    const figureMap = {};



    // Map each figure's ID to its number
    figures.forEach((figure, index) => {
        const id = figure.id; // Get the ID of each figure
        if (id) {
            figureMap[id] = index + 1;

        }
    });

    // Process all the links
    const links = document.querySelectorAll("body .figure-link");


    links.forEach(link => {
        const targetId = link.getAttribute("data-target");


        const figureNumber = figureMap[targetId];
        if (figureNumber !== undefined) {
            link.href = `#${targetId}`;
            link.textContent = `Figure ${figureNumber}`;

        }
    });
});