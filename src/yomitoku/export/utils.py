def sort_elements(elements, directions):
    if directions.count("vertical") > directions.count("horizontal"):
        return sorted(elements, key=lambda x: x["box"][0], reverse=True)
    return sorted(elements, key=lambda x: x["box"][1])
