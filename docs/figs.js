document.addEventListener("DOMContentLoaded", () => {
    // Select <figure> elements inside <d-article> and <d-appendix>, excluding those inside <d-title>
    const articles = document.querySelectorAll("d-article");
    const appendices = document.querySelectorAll("d-appendix");
    const figureMap = {};

    // Helper function to map figures in a given section
    function mapFigures(section, startIndex) {
        const figures = section.querySelectorAll("figure");
        figures.forEach((figure, index) => {
            const id = figure.id;
            if (id) {
                figureMap[id] = startIndex + index + 1;
            }
        });
        return startIndex + figures.length;
    }

    // Map figures in articles
    let figureIndex = 0;
    articles.forEach(article => {
        figureIndex = mapFigures(article, figureIndex);
    });

    // Map figures in appendices (resetting the counter)
    figureIndex = 0;
    appendices.forEach(appendix => {
        figureIndex = mapFigures(appendix, figureIndex);
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