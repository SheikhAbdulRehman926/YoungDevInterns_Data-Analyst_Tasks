# Power BI Dashboard Plan

This guide outlines how to build an interactive dashboard in Power BI using the cleaned dataset produced by `analysis/advanced_analysis.py`.

## 1. Connect to data
1. Run `python analysis/advanced_analysis.py` to create `outputs/clean_retail_kpis.csv`.
2. In Power BI Desktop select **Get Data → Text/CSV → outputs/clean_retail_kpis.csv**.
3. Confirm data types:
   - `Date` → Date.
   - `Region`, `Channel` → Categorical.
   - Metrics (`Sales`, `MarketingSpend`, `SalesPerCustomer`, etc.) → Decimal numbers.

## 2. Model relationships and calculations
- Create a **Date table** (`Modeling → New table`) with `CALENDAR (DATE(2024,1,1), DATE(2024,12,31))`.
- Relate `Date[Date]` to `clean_retail_kpis[Date]`.
- Suggested DAX measures:
  - `Total Sales = SUM(clean_retail_kpis[Sales])`
  - `Total Marketing Spend = SUM(clean_retail_kpis[MarketingSpend])`
  - `Sales per Customer = DIVIDE(SUM(clean_retail_kpis[Sales]), SUM(clean_retail_kpis[TotalCustomers]))`
  - `YoY Sales Growth = VAR Curr = CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date])) RETURN DIVIDE([Total Sales] - Curr, Curr)`

## 3. Visual layout
1. **Executive KPI strip**: Cards for Total Sales, Marketing Spend, MAE from model metrics (manual entry), and YoY growth.
2. **Sales trend**: Line chart with `Date` on axis and `Total Sales` as value; add slicers for `Region` and `Channel`.
3. **Funnel from marketing to revenue**: Stacked column chart showing `MarketingSpend` vs `Sales` by `Region`.
4. **Customer mix**: 100% stacked bar chart with `Region` on axis and `NewCustomerRate` vs `ReturningCustomers`.
5. **Scatter analysis**: Scatter plot with `MarketingSpend` (X), `Sales` (Y), bubble size `WebTraffic`, legend `Channel`.
6. **Map (optional)**: Use shape map or filled map to show regional performance; map regions to custom shapes if needed.

## 4. Interactivity
- Enable **cross-highlighting** on the scatter plot so selecting a channel updates all visuals.
- Add a **drill-through page** filtered by `Region` with detailed monthly metrics and a decomposition tree (Sales → Channel → Customer type).
- Create bookmarks for *Executive summary*, *Marketing focus*, and *Customer insights* scenarios, then wire them to buttons for guided storytelling.

## 5. Publish and share
1. Validate refresh by selecting **Refresh**; confirm that transformations are applied correctly.
2. Publish to the Power BI Service workspace dedicated to analytics prototypes.
3. Configure a refresh schedule pointing to the CSV location (or migrate the dataset to a SharePoint/OneDrive folder).
4. Document data lineage and provide usage instructions inside the workspace description.

Following these steps yields an interactive dashboard aligned with the analytical findings and predictive model outputs in this repository.

